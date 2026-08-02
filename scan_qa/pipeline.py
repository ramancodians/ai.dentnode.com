"""Scan QA pipeline.

    STL / PLY  →  cleanup  →  segmentation  →  per-tooth + arch QC  →  score
                                                                        ↓
                                                          LLM writes the prose

The model never decides. `overall_score` and `risk_level` come from the
deterministic checks in `checks.py`; the LLM layer downstream only explains
findings that geometry already established.

Reuses `scan_review.geometry` for mesh-level measurement (holes, non-manifold
edges, winding) rather than reimplementing it — this module adds the
segmentation-conditioned layer on top.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from scan_review.geometry import inspect_mesh

from .checks import (
    CRITICAL,
    INFO,
    WARNING,
    Finding,
    check_adjacent_teeth,
    check_bite_coverage,
    check_excess_soft_tissue,
    check_holes,
    check_opposing_arch,
    check_scan_islands,
    check_spikes_and_stitching,
    check_tooth_capture,
)
from .segmentation import Segmentation, Segmenter

logger = logging.getLogger(__name__)

_SEVERITY_PENALTY = {INFO: 0, WARNING: 12, CRITICAL: 45}
# With any CRITICAL present the score is held at or below this, so the number
# can never read "minor issues" beside a critical finding.
_CRITICAL_CAP = 60.0

# How much each role contributes to the case score. Deliberately NOT a minimum
# across files: a ruined bite registration is a real problem but it does not
# make a pair of flawless arch scans worthless, and scoring it that way trains
# technicians to ignore the number. The prepared arch dominates because it is
# the file the restoration is actually built on.
_ROLE_WEIGHT = {"prepared_arch": 0.5, "opposing_arch": 0.2, "bite": 0.3}
_UNWEIGHTED_ROLE = 0.2


@dataclass
class ScanRole:
    """One input file and what it is for. Roles drive which checks apply —
    judging a buccal bite by the rubric for a prepared arch is the single
    biggest source of false alarms."""

    label: str
    path_or_mesh: Any
    role: str  # "prepared_arch" | "opposing_arch" | "bite"
    jaw: str = "upper"


@dataclass
class FileQA:
    label: str
    role: str
    ok: bool = True
    error: Optional[str] = None
    geometry: Optional[Dict[str, Any]] = None
    segmentation: Optional[Dict[str, Any]] = None
    findings: List[Finding] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "role": self.role,
            "ok": self.ok,
            "error": self.error,
            "geometry": self.geometry,
            "segmentation": self.segmentation,
            "findings": [f.as_dict() for f in self.findings],
        }


@dataclass
class ScanQAReport:
    overall_score: float
    risk_level: str
    files: List[FileQA]
    findings: List[Finding]
    elapsed_ms: int
    summary: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 1),
            "risk_level": self.risk_level,
            "summary": self.summary,
            "findings": [f.as_dict() for f in self.findings],
            "files": [f.as_dict() for f in self.files],
            "elapsed_ms": self.elapsed_ms,
        }


def _load(item: Any) -> trimesh.Trimesh:
    if isinstance(item, trimesh.Trimesh):
        return item
    mesh = trimesh.load(item, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"not a single mesh: {item}")
    return mesh


def _measure(item: Any, label: str) -> Dict[str, Any]:
    """Mesh-level measurement via scan_review.geometry.

    That function takes raw bytes plus a file type (it owns its own loading and
    the STL vertex-merge), so a path is read back rather than re-exported — an
    export round-trip would change the very topology being measured.
    """
    from pathlib import Path as _Path

    if isinstance(item, (str, _Path)):
        p = _Path(item)
        return inspect_mesh(
            p.read_bytes(), p.suffix.lstrip(".").lower(), label=label
        ).as_dict()

    export = item.export(file_type="ply")
    if isinstance(export, str):
        export = export.encode()
    return inspect_mesh(export, "ply", label=label).as_dict()


def _clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalisation every downstream step assumes.

    STL is triangle soup — vertices are not shared between faces, so without a
    merge every edge reads as a boundary and topology checks are meaningless.
    """
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def _role_score(findings: List[Finding]) -> float:
    """0-100 for one role. Repeat findings of a kind are capped so that a file
    with 700 fragments is not scored 60 points worse than one with 20 — both
    need the same action."""
    by_type: Dict[str, str] = {}
    order = {INFO: 0, WARNING: 1, CRITICAL: 2}
    for f in findings:
        if order.get(f.severity, 0) > order.get(by_type.get(f.type, INFO), 0):
            by_type[f.type] = f.severity
        by_type.setdefault(f.type, f.severity)
    penalty = sum(_SEVERITY_PENALTY.get(sev, 0) for sev in by_type.values())
    return max(0.0, 100.0 - penalty)


def _score(files: List["FileQA"], case_findings: List[Finding]) -> Tuple[float, str]:
    """Weighted across roles rather than a minimum across files."""
    per_role: Dict[str, List[Finding]] = {}
    for fq in files:
        per_role.setdefault(fq.role, []).extend(fq.findings)
    for f in case_findings:
        # Bite/opposing case-level findings are attributed to their own role so
        # they are weighted, not applied flat to the whole case.
        role = (
            "bite"
            if "bite" in f.type
            else "opposing_arch"
            if "opposing" in f.type
            else "prepared_arch"
        )
        per_role.setdefault(role, []).append(f)

    total_w = 0.0
    acc = 0.0
    for role, findings in per_role.items():
        w = _ROLE_WEIGHT.get(role, _UNWEIGHTED_ROLE)
        acc += w * _role_score(findings)
        total_w += w
    score = acc / total_w if total_w else 0.0

    all_findings = [f for fs in per_role.values() for f in fs]
    has_critical = any(f.severity == CRITICAL for f in all_findings)
    if has_critical:
        score = min(score, _CRITICAL_CAP)
    if has_critical or score < 50:
        risk = "HIGH"
    elif score < 80:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return score, risk


def run_scan_qa(
    scans: List[ScanRole],
    *,
    segmenter: Optional[Segmenter] = None,
    prep_teeth: Optional[List[int]] = None,
) -> ScanQAReport:
    """Run the full QA pass over one case."""
    t0 = time.monotonic()
    prep_teeth = prep_teeth or []
    files: List[FileQA] = []
    meshes: Dict[str, trimesh.Trimesh] = {}
    segs: Dict[str, Segmentation] = {}

    # -- per-file: load, measure, segment, check ----------------------------
    for scan in scans:
        fq = FileQA(label=scan.label, role=scan.role)
        prefix = scan.label.lower().replace(" ", "_")
        try:
            mesh = _clean(_load(scan.path_or_mesh))
        except Exception as exc:  # noqa: BLE001
            fq.ok = False
            fq.error = str(exc)
            fq.findings.append(
                Finding(
                    id=f"{prefix}-unreadable",
                    type="unreadable_file",
                    severity=CRITICAL,
                    title=f"Could not read '{scan.label}'",
                    detail=str(exc),
                    recommendation="Re-upload the file.",
                )
            )
            files.append(fq)
            continue

        meshes[scan.role] = mesh
        fq.geometry = _measure(scan.path_or_mesh, scan.label)

        # Mesh-level checks apply to every role.
        fq.findings += check_holes(fq.geometry, prefix, scan.label)
        fq.findings += check_scan_islands(mesh, prefix, scan.label)
        fq.findings += check_spikes_and_stitching(mesh, prefix, scan.label)

        # Segmentation-dependent checks only make sense on an arch. A bite
        # registration has no arch form to segment into teeth.
        if scan.role in ("prepared_arch", "opposing_arch") and segmenter is not None:
            try:
                seg = segmenter.segment(mesh, jaw=scan.jaw)
                segs[scan.role] = seg
                fq.segmentation = seg.as_dict()
                fq.findings += check_excess_soft_tissue(seg, prefix)
                fq.findings += check_tooth_capture(seg, prefix)
                if scan.role == "prepared_arch":
                    fq.findings += check_adjacent_teeth(seg, prep_teeth, prefix)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Segmentation failed", extra={"label": scan.label})
                fq.findings.append(
                    Finding(
                        id=f"{prefix}-seg-failed",
                        type="segmentation_failed",
                        severity=INFO,
                        title="Segmentation unavailable",
                        detail=f"{exc}. Mesh-level checks still ran.",
                        recommendation="No action; per-tooth attribution is skipped.",
                    )
                )
        files.append(fq)

    # -- case-level checks ---------------------------------------------------
    case_findings: List[Finding] = []
    prepared = meshes.get("prepared_arch")
    if prepared is not None:
        case_findings += check_opposing_arch(
            prepared, meshes.get("opposing_arch"), "case"
        )
    bites = [
        (s.label, meshes[s.role])
        for s in scans
        if s.role == "bite" and s.role in meshes
    ]
    # meshes is keyed by role, so multiple bites collapse; re-collect properly.
    bites = []
    for scan in scans:
        if scan.role != "bite":
            continue
        try:
            bites.append((scan.label, _clean(_load(scan.path_or_mesh))))
        except Exception:  # noqa: BLE001
            pass
    case_findings += check_bite_coverage(bites, "case")

    all_findings = [f for fq in files for f in fq.findings] + case_findings
    score, risk = _score(files, case_findings)

    return ScanQAReport(
        overall_score=score,
        risk_level=risk,
        files=files,
        findings=all_findings,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )
