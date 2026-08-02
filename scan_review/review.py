"""Scan Review orchestration: fetch → measure → narrate.

The pipeline for one request:

    URLs ──fetch.py──> bytes ──geometry.py──> measurements + findings + score
                                                        │
                                                        ▼
                                              LLM narrative (prose only)

The split matters. **Geometry decides, the model describes.** `overall_score`
and `risk_level` come from `geometry.py` and nothing the model says can lower
them — a QA gate has to be reproducible, and two runs over the same STL must
agree. The model's job is to turn "3 boundary loops, largest 4.2 mm²" into a
sentence a lab technician acts on. It is allowed to *raise* risk (it may spot a
combination the thresholds miss) but never to lower it.

If the model call fails the request still succeeds: the measurements are the
valuable part and they are already in hand. `llm_error` records what happened.

**Shared infrastructure, deliberately.** This module is standalone — it has no
ADK, no Laby tools, no agent session — but it does use the service's existing
OpenRouter gateway and usage ledger client rather than reimplementing them.
A second metering path would be a real risk to billing correctness, and there
is exactly one OpenRouter account.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.openrouter import OpenRouterError, chat_completion

from .config import settings
from .fetch import fetch_all
from .geometry import (
    MeshInspectError,
    MeshReport,
    build_findings,
    inspect_mesh,
    risk_from_findings,
    score_report,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the DentNode Scan Review, a dental laboratory 3D file QA reviewer.

You are given MEASURED geometry from one or more scan files (STL/PLY/OBJ). These
numbers were computed deterministically from the mesh — they are facts, not
estimates. You cannot see the scan; do not claim to.

Your job is to explain what the measurements mean for this specific case, in the
language of a dental lab. Be concrete and brief. A technician should be able to
read your summary and know whether to proceed, repair, or request a rescan.

Key domain rules:
- A raw intraoral scan is an OPEN surface. Its outer trim boundary is normal and
  is reported separately as "trim_boundary" — never call that a defect.
- "holes" are interior boundary loops. Those ARE defects. A hole on a margin,
  prep, or occlusal surface is far worse than one on a gingival flank.
- Non-manifold edges break CAM/slicer software. Treat them as blocking.
- An implausible bounding box almost always means wrong export units, not a
  genuinely giant model.

Return ONLY valid JSON — no markdown, no prose outside the JSON object.

Schema:
{
  "summary": "<2-4 sentences: overall verdict and what to do next>",
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "findings": [
    {
      "id": "a1",
      "category": "MESH_QUALITY" | "COMPLETENESS" | "PRINT_READINESS" | "DIMENSIONS" | "WORKFLOW",
      "severity": "INFO" | "WARNING" | "CRITICAL",
      "title": "<short title>",
      "detail": "<what the measurements show, and why it matters for this case>",
      "recommendation": "<what the lab should do>"
    }
  ],
  "flags": ["<short_snake_case_tags>"]
}

Do not restate every number back. Add interpretation the raw measurements do not
already give: what it means for this product type, and what the lab should do."""


class ScanReviewError(RuntimeError):
    """Raised only when the whole request cannot proceed."""


@dataclass
class FileResult:
    """Outcome for one requested file — measured, or the reason it wasn't."""

    label: str
    url: str
    ok: bool
    report: Optional[MeshReport] = None
    findings: List[Dict[str, Any]] = field(default_factory=list)
    score: Optional[float] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "url": self.url,
            "ok": self.ok,
            "score": self.score,
            "geometry": self.report.as_dict() if self.report else None,
            "findings": self.findings,
            "error": self.error,
        }


@dataclass
class ScanReviewResult:
    summary: Optional[str]
    risk_level: str
    overall_score: Optional[float]
    findings: List[Dict[str, Any]]
    flags: List[str]
    files: List[FileResult]
    model: Optional[str] = None
    parsed: bool = False
    raw_response: str = ""
    llm_error: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: Optional[float] = None
    latency_ms: int = 0

    @property
    def analyzed_count(self) -> int:
        return sum(1 for f in self.files if f.ok)


# ── Parsing ───────────────────────────────────────────────────────────────────


def parse_model_json(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON object from a model reply.

    Returns None rather than raising: the narrative is a bonus layer on top of
    measurements that already succeeded, so an unparseable reply degrades to
    "no prose" instead of failing the review.
    """
    cleaned = raw or ""
    cleaned = re.sub(r"^```json\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:  # noqa: BLE001
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:  # noqa: BLE001
        return None


# ── Aggregation ───────────────────────────────────────────────────────────────

_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _escalate_only(measured: str, suggested: Any) -> str:
    """Let the model raise risk, never lower it.

    The measured risk is derived from thresholds over real geometry. A model
    that talks itself into "LOW" on a mesh with 40 non-manifold edges must not
    be able to wave the case through.
    """
    if not isinstance(suggested, str):
        return measured
    candidate = suggested.strip().upper()
    if candidate not in _RISK_ORDER:
        return measured
    return candidate if _RISK_ORDER[candidate] > _RISK_ORDER[measured] else measured


def build_flags(files: List[FileResult]) -> List[str]:
    """Deterministic short tags, useful for filtering and dashboards."""
    flags: set = set()
    for f in files:
        if not f.ok or f.report is None:
            flags.add("file_unreadable")
            continue
        r = f.report
        if r.holes:
            flags.add("holes")
            if any(h.area_mm2 >= 1.0 for h in r.holes):
                flags.add("large_holes")
        if r.non_manifold_edges:
            flags.add("non_manifold")
        if not r.winding_consistent:
            flags.add("inconsistent_normals")
        if r.fragments:
            flags.add("floating_fragments")
        if r.degenerate_faces:
            flags.add("degenerate_faces")
        if r.expect_watertight and not r.watertight:
            flags.add("not_watertight")
        if r.bounding_box_mm:
            largest = max(r.bounding_box_mm)
            if largest < 3.0 or largest > 250.0:
                flags.add("scale_suspect")
    if not flags:
        flags.add("clean")
    return sorted(flags)


# How many individual hole / fragment records to send per file. A noisy bite
# scan can carry thousands; serialising them all produced a 335k-token prompt
# (real case: 3,436 holes over 4 files) which no model will accept and which
# tells the model nothing the counts and the worst offenders do not. The counts
# themselves are always sent in full — only the per-record detail is trimmed.
_MAX_DETAIL_RECORDS = 20


def _compact_geometry(report: Dict[str, Any]) -> Dict[str, Any]:
    """Trim unbounded per-record lists down to the worst offenders.

    Keeps the largest holes and fragments, because those are the ones that
    decide whether a scan is usable; the rest are represented by a count so the
    model still knows the true scale.
    """
    compact = dict(report)

    holes = compact.get("holes")
    if isinstance(holes, list) and len(holes) > _MAX_DETAIL_RECORDS:
        ranked = sorted(
            holes, key=lambda h: h.get("area_mm2") or 0.0, reverse=True
        )
        compact["holes"] = ranked[:_MAX_DETAIL_RECORDS]
        compact["holes_shown"] = _MAX_DETAIL_RECORDS
        compact["holes_omitted"] = len(holes) - _MAX_DETAIL_RECORDS
        compact["holes_note"] = (
            f"{len(holes)} holes total; the {_MAX_DETAIL_RECORDS} largest by "
            "area are listed."
        )

    fragments = compact.get("fragments")
    if isinstance(fragments, list) and len(fragments) > _MAX_DETAIL_RECORDS:
        ranked = sorted(
            fragments, key=lambda f: f.get("faces") or 0, reverse=True
        )
        compact["fragments"] = ranked[:_MAX_DETAIL_RECORDS]
        compact["fragments_shown"] = _MAX_DETAIL_RECORDS
        compact["fragments_omitted"] = len(fragments) - _MAX_DETAIL_RECORDS

    return compact


def _build_user_content(
    *, case_context: Dict[str, Any], files: List[FileResult], today: str
) -> str:
    """Serialise the measurements for the model.

    Sends the structured report rather than prose so the model interprets
    numbers instead of re-deriving them from a description.
    """
    payload = {
        "today": today,
        "case_context": case_context,
        "files": [
            {
                "label": f.label,
                "analyzed": f.ok,
                "error": f.error,
                "measured_score": f.score,
                "geometry": _compact_geometry(f.report.as_dict()) if f.report else None,
                "geometry_findings": f.findings,
            }
            for f in files
        ],
    }
    return (
        "Measured geometry for this case follows as JSON.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n\nReview it against the case context and return the JSON object "
        "described in your instructions."
    )


# ── Entry point ───────────────────────────────────────────────────────────────


async def _analyze_one(spec: Dict[str, Any], mesh, error: Optional[str]) -> FileResult:
    """Measure a single fetched mesh, off the event loop."""
    label = spec.get("label") or "scan"
    url = spec.get("url", "")

    if error is not None or mesh is None:
        return FileResult(label=label, url=url, ok=False, error=error or "not fetched")

    try:
        report = await asyncio.wait_for(
            asyncio.to_thread(
                inspect_mesh,
                mesh.data,
                mesh.file_type,
                label=label,
                expect_watertight=spec.get("expect_watertight"),
            ),
            timeout=float(settings.analyze_timeout_secs),
        )
    except asyncio.TimeoutError:
        logger.warning("Mesh inspection timed out", extra={"label": label})
        return FileResult(
            label=label,
            url=url,
            ok=False,
            error=f"Mesh inspection exceeded {settings.analyze_timeout_secs}s",
        )
    except MeshInspectError as exc:
        return FileResult(label=label, url=url, ok=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - a bad mesh must not 500 the request
        logger.exception("Mesh inspection failed", extra={"label": label})
        return FileResult(
            label=label, url=url, ok=False, error=f"Inspection failed: {exc}"
        )

    return FileResult(
        label=label,
        url=url,
        ok=True,
        report=report,
        findings=build_findings(report),
        score=score_report(report),
    )


async def generate_scan_review(
    *,
    files: List[Dict[str, Any]],
    case_context: Optional[Dict[str, Any]] = None,
    today: str,
    use_llm: Optional[bool] = None,
) -> ScanReviewResult:
    """Review every supplied mesh URL and return one combined QA result.

    Raises ScanReviewError only for a request-level problem (no files, too many
    files). A single bad URL or unparseable mesh becomes a failed FileResult and
    the rest of the case is still reviewed.
    """
    case_context = case_context or {}

    if not files:
        raise ScanReviewError("At least one file is required")
    if len(files) > settings.max_files:
        raise ScanReviewError(
            f"Too many files: {len(files)} (limit {settings.max_files}, "
            "SCAN_REVIEW_MAX_FILES)"
        )

    t0 = time.monotonic()

    fetched = await fetch_all(files)
    results = await asyncio.gather(
        *(_analyze_one(spec, mesh, err) for spec, mesh, err in fetched)
    )
    file_results: List[FileResult] = list(results)

    geometry_findings: List[Dict[str, Any]] = []
    for f in file_results:
        geometry_findings.extend(f.findings)
        if not f.ok:
            geometry_findings.append(
                {
                    "id": f"{f.label}-unreadable",
                    "category": "COMPLETENESS",
                    "severity": "CRITICAL",
                    "title": f"Could not analyse '{f.label}'",
                    "detail": f.error or "The file could not be fetched or parsed.",
                    "recommendation": "Re-upload the file or check the link, then re-run the review.",
                    "source": "geometry",
                }
            )

    scored = [f.score for f in file_results if f.score is not None]
    # The case is only as good as its worst file — averaging would let a clean
    # lower arch hide a holed upper.
    measured_score = min(scored) if scored else 0.0
    measured_risk = risk_from_findings(geometry_findings, measured_score)
    flags = build_flags(file_results)

    result = ScanReviewResult(
        summary=None,
        risk_level=measured_risk,
        overall_score=measured_score,
        findings=list(geometry_findings),
        flags=flags,
        files=file_results,
    )

    run_llm = settings.enable_llm if use_llm is None else use_llm
    if not run_llm:
        result.summary = _fallback_summary(file_results, measured_score)
        return result

    try:
        completion = await chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_content(
                        case_context=case_context, files=file_results, today=today
                    ),
                },
            ],
            model=settings.model,
            temperature=0.2,
            max_tokens=2048,
            timeout_secs=float(settings.llm_timeout_secs),
        )
    except OpenRouterError as exc:
        # Measurements already succeeded; a model outage must not lose them.
        logger.warning("Scan review narrative failed", extra={"error": str(exc)})
        result.llm_error = str(exc)
        result.summary = _fallback_summary(file_results, measured_score)
        result.latency_ms = int((time.monotonic() - t0) * 1000)
        return result

    result.raw_response = completion.text
    result.model = completion.model
    result.usage = completion.usage
    result.cost_usd = completion.cost_usd
    result.latency_ms = int((time.monotonic() - t0) * 1000)

    parsed = parse_model_json(completion.text)
    if parsed is None:
        logger.warning(
            "Scan review narrative was not parseable JSON",
            extra={"chars": len(completion.text)},
        )
        result.summary = _fallback_summary(file_results, measured_score)
        return result

    result.parsed = True
    summary = parsed.get("summary")
    result.summary = summary if isinstance(summary, str) and summary.strip() else (
        _fallback_summary(file_results, measured_score)
    )
    result.risk_level = _escalate_only(measured_risk, parsed.get("risk_level"))

    # Model findings are appended after the measured ones and tagged, so the UI
    # can always tell which claims are backed by geometry.
    model_findings = parsed.get("findings")
    if isinstance(model_findings, list):
        for item in model_findings:
            if isinstance(item, dict):
                result.findings.append({**item, "source": "ai"})

    model_flags = parsed.get("flags")
    if isinstance(model_flags, list):
        merged = set(result.flags) | {
            str(f) for f in model_flags if isinstance(f, (str, int))
        }
        # "clean" is only meaningful on its own.
        if len(merged) > 1:
            merged.discard("clean")
        result.flags = sorted(merged)

    return result


def _fallback_summary(files: List[FileResult], score: float) -> str:
    """Plain-language summary built from measurements only.

    Used when the narrative layer is disabled, fails, or returns junk — the
    response should never come back without a readable verdict.
    """
    ok = [f for f in files if f.ok and f.report]
    failed = [f for f in files if not f.ok]

    if not ok:
        return (
            f"None of the {len(files)} supplied file(s) could be analysed. "
            "Check the links and re-run the review."
        )

    holes = sum(len(f.report.holes) for f in ok if f.report)
    non_manifold = sum(f.report.non_manifold_edges for f in ok if f.report)
    fragments = sum(len(f.report.fragments) for f in ok if f.report)

    parts = [f"Analysed {len(ok)} file(s); QA score {score:.0f}/100."]
    if holes:
        parts.append(f"{holes} interior hole(s) found.")
    if non_manifold:
        parts.append(f"{non_manifold} non-manifold edge(s).")
    if fragments:
        parts.append(f"{fragments} floating fragment(s).")
    if not (holes or non_manifold or fragments):
        parts.append("No hole, manifold, or fragment defects detected.")
    if failed:
        parts.append(f"{len(failed)} file(s) could not be read.")

    return " ".join(parts)
