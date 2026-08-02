"""Deterministic QC checks for Scan QA.

Every check here is an explicit geometric algorithm with a stated threshold. The
neural network only says *where* things are; nothing in this file asks a model
whether a scan is good. That split is the point of the design: a segmentation
error degrades attribution ("which tooth"), it does not silently flip a verdict.

Each check returns zero or more `Finding`s. Thresholds live at the top of their
check so they can be tuned against technician feedback without hunting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh

from .segmentation import GINGIVA, Segmentation

INFO, WARNING, CRITICAL = "INFO", "WARNING", "CRITICAL"


@dataclass
class Location:
    """Where a finding sits on the mesh, in the scan's own coordinates (mm).

    Carried so a 3-D viewer can fly the camera to a finding and mark it, rather
    than leaving the technician to hunt for "a hole somewhere on the bite scan".
    `radius_mm` sizes the marker; None means a point rather than an extent.
    """

    x: float
    y: float
    z: float
    radius_mm: Optional[float] = None

    @classmethod
    def from_xyz(cls, xyz, radius_mm: Optional[float] = None) -> "Location":
        return cls(
            float(xyz[0]), float(xyz[1]), float(xyz[2]),
            None if radius_mm is None else float(radius_mm),
        )

    def as_dict(self) -> Dict[str, Any]:
        d = {"x": round(self.x, 3), "y": round(self.y, 3), "z": round(self.z, 3)}
        if self.radius_mm is not None:
            d["radius_mm"] = round(self.radius_mm, 3)
        return d


@dataclass
class Finding:
    id: str
    type: str
    severity: str
    title: str
    detail: str
    recommendation: str
    tooth: Optional[int] = None
    measurements: Dict[str, Any] = field(default_factory=dict)
    # Which file this sits on, so the viewer can select the right model.
    file_label: Optional[str] = None
    # Primary location plus any extra spots of the same kind (e.g. every hole).
    location: Optional[Location] = None
    locations: List[Location] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "recommendation": self.recommendation,
            "tooth": self.tooth,
            "measurements": self.measurements,
            "file_label": self.file_label,
            "location": self.location.as_dict() if self.location else None,
            "locations": [l.as_dict() for l in self.locations],
            "source": "geometry",
        }


# ── 0. Interior holes ─────────────────────────────────────────────────────────

# A void larger than this is a region the scanner never captured. Filling it
# invents surface that was never measured, which is unsafe on a margin or an
# occlusal contact — so it is escalated rather than treated as a patchable defect.
_LARGE_HOLE_MM2 = 25.0
# How many individual hole locations to hand the viewer. Marking 1,600 holes
# renders as noise; the largest few are what a technician navigates to.
_MAX_HOLE_MARKERS = 25


def check_holes(geometry: Dict[str, Any], prefix: str, label: str) -> List[Finding]:
    """Interior boundary loops, i.e. holes that are not the scan's trim edge.

    `geometry` is a `scan_review.geometry.MeshReport.as_dict()`. The trim
    boundary is already classified there and deliberately excluded — the open
    edge of an arch scan is expected, not a defect.
    """
    holes = geometry.get("holes") or []
    if not holes:
        return []

    ranked = sorted(holes, key=lambda h: h.get("area_mm2") or 0.0, reverse=True)
    large = [h for h in ranked if (h.get("area_mm2") or 0.0) > _LARGE_HOLE_MM2]
    biggest = ranked[0]

    locations = [
        Location.from_xyz(
            h["centroid"], (h.get("max_diameter_mm") or 0.0) / 2.0
        )
        for h in ranked[:_MAX_HOLE_MARKERS]
        if h.get("centroid")
    ]

    detail = (
        f"{len(holes)} closed boundary loop(s) inside the mesh. The largest spans "
        f"{biggest.get('area_mm2', 0):.2f} mm² "
        f"({biggest.get('max_diameter_mm', 0):.2f} mm across)."
    )
    if large:
        detail += (
            f" {len(large)} exceed {_LARGE_HOLE_MM2:.0f} mm² — those are regions the "
            "scanner never captured, not defects to patch."
        )

    return [
        Finding(
            id=f"{prefix}-holes",
            type="missing_scan_area" if large else "surface_holes",
            severity=CRITICAL if large else WARNING,
            title=f"{len(holes)} hole(s) in the surface",
            detail=detail,
            recommendation=(
                "Rescan the affected area; do not fill a void this size."
                if large
                else "Close the small holes during clean-up."
            ),
            file_label=label,
            location=locations[0] if locations else None,
            locations=locations,
            measurements={
                "holes": len(holes),
                "large_holes": len(large),
                "largest_area_mm2": round(biggest.get("area_mm2") or 0.0, 2),
            },
        )
    ]


# ── 1. Scan islands / floating geometry ───────────────────────────────────────

# A connected component under this share of total area is debris, not anatomy.
_ISLAND_AREA_FRACTION = 0.02


# Markers handed to the viewer for fragments — the biggest ones, since those are
# the ones worth navigating to.
_MAX_ISLAND_MARKERS = 25


def check_scan_islands(
    mesh: trimesh.Trimesh, prefix: str, label: Optional[str] = None
) -> List[Finding]:
    """Disconnected fragments — captured tongue, cheek, saliva, scanner noise."""
    comps = mesh.split(only_watertight=False)
    if len(comps) <= 1:
        return []

    total = float(mesh.area)
    islands = [c for c in comps if float(c.area) / total < _ISLAND_AREA_FRACTION]
    if not islands:
        return []

    area = sum(float(c.area) for c in islands)
    faces = sum(len(c.faces) for c in islands)
    sev = CRITICAL if len(islands) > 200 else WARNING

    biggest = sorted(islands, key=lambda c: float(c.area), reverse=True)
    locations = [
        Location.from_xyz(
            c.centroid, float(np.linalg.norm(c.extents)) / 2.0
        )
        for c in biggest[:_MAX_ISLAND_MARKERS]
    ]

    return [
        Finding(
            id=f"{prefix}-islands",
            type="scan_islands",
            severity=sev,
            title=f"{len(islands)} floating fragment(s)",
            detail=(
                f"{len(islands)} disconnected piece(s) totalling {faces} faces "
                f"({area:.1f} mm²) sit apart from the main surface."
            ),
            recommendation="Delete stray fragments before design.",
            file_label=label,
            location=locations[0] if locations else None,
            locations=locations,
            measurements={"islands": len(islands), "area_mm2": round(area, 2)},
        )
    ]


# ── 2. Surface spikes and stitching artefacts ─────────────────────────────────

# A vertex whose incident faces disagree in direction beyond this is a spike,
# not curvature. Real enamel curvature stays well under it at scan resolution.
_SPIKE_NORMAL_DEG = 110.0
# Edges this many times the median are stitching stretch across a gap.
_STRETCH_EDGE_FACTOR = 8.0


_MAX_SPIKE_MARKERS = 25


def check_spikes_and_stitching(
    mesh: trimesh.Trimesh, prefix: str, label: Optional[str] = None
) -> List[Finding]:
    """Single-vertex spikes and over-long stretched triangles."""
    out: List[Finding] = []

    fn = mesh.face_normals
    adj = mesh.face_adjacency
    if len(adj):
        dots = np.einsum("ij,ij->i", fn[adj[:, 0]], fn[adj[:, 1]])
        angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        spike_mask = angles > _SPIKE_NORMAL_DEG
        spikes = int(spike_mask.sum())
        spike_locs: List[Location] = []
        if spikes:
            # Sharpest first — the worst artefacts are the ones to inspect.
            worst = np.argsort(-angles[spike_mask])[:_MAX_SPIKE_MARKERS]
            pairs = adj[spike_mask][worst]
            centres = mesh.triangles_center
            spike_locs = [
                Location.from_xyz((centres[a] + centres[b]) / 2.0)
                for a, b in pairs
            ]
        if spikes:
            sev = CRITICAL if spikes > len(mesh.faces) * 0.005 else WARNING
            out.append(
                Finding(
                    id=f"{prefix}-spikes",
                    type="surface_spikes",
                    severity=sev,
                    title=f"{spikes} spike / fold artefact(s)",
                    detail=(
                        f"{spikes} adjacent face pair(s) meet at more than "
                        f"{_SPIKE_NORMAL_DEG:.0f}°, which is a scanner spike or a "
                        "folded-over surface rather than real anatomy."
                    ),
                    recommendation=(
                        "Run spike removal / smoothing before design; a spike on "
                        "an occlusal surface becomes a high spot on the restoration."
                    ),
                    file_label=label,
                    location=spike_locs[0] if spike_locs else None,
                    locations=spike_locs,
                    measurements={"spike_pairs": spikes},
                )
            )

    edges = mesh.vertices[mesh.edges_unique]
    lengths = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1)
    if len(lengths):
        median = float(np.median(lengths))
        stretched = int((lengths > median * _STRETCH_EDGE_FACTOR).sum())
        if stretched:
            out.append(
                Finding(
                    id=f"{prefix}-stretch",
                    type="stitching_artifact",
                    severity=WARNING,
                    title=f"{stretched} stretched triangle edge(s)",
                    detail=(
                        f"{stretched} edge(s) exceed {_STRETCH_EDGE_FACTOR:.0f}x the "
                        f"median edge ({median:.3f} mm) — surface stretched across a "
                        "gap during stitching rather than measured."
                    ),
                    recommendation=(
                        "Treat these spans as unmeasured. Rescan if any crosses a "
                        "margin or occlusal contact."
                    ),
                    measurements={
                        "stretched_edges": stretched,
                        "median_edge_mm": round(median, 4),
                    },
                )
            )
    return out


# ── 3. Excess soft tissue ─────────────────────────────────────────────────────

# Above this share of the surface, the scan is mostly tissue and the operator
# has over-captured cheek/palate rather than dentition.
_GINGIVA_HIGH = 0.72


def check_excess_soft_tissue(seg: Segmentation, prefix: str) -> List[Finding]:
    frac = seg.gingiva_fraction
    if frac < _GINGIVA_HIGH:
        return []
    return [
        Finding(
            id=f"{prefix}-soft-tissue",
            type="excess_soft_tissue",
            severity=WARNING,
            title=f"{frac*100:.0f}% of the scan is soft tissue",
            detail=(
                f"Segmentation attributes {frac*100:.0f}% of the surface to gingiva "
                "and mucosa rather than teeth."
            ),
            recommendation=(
                "Trim excess tissue. Large mucosa areas slow design and can "
                "confuse automatic margin detection."
            ),
            measurements={"gingiva_fraction": round(frac, 3)},
        )
    ]


# ── 4. Adjacent-tooth support ─────────────────────────────────────────────────


def check_adjacent_teeth(
    seg: Segmentation, prep_teeth: List[int], prefix: str
) -> List[Finding]:
    """Each prep needs a neighbour captured to build contacts against.

    Only runs when FDI mapping is available; without it "adjacent" cannot be
    tied to the tooth numbers the case is written in.
    """
    detected = {t.fdi for t in seg.teeth if t.fdi is not None}
    if not detected or not prep_teeth:
        return []

    out: List[Finding] = []
    for tooth in prep_teeth:
        neighbours = {tooth - 1, tooth + 1}
        if neighbours & detected:
            continue
        out.append(
            Finding(
                id=f"{prefix}-adj-{tooth}",
                type="insufficient_adjacent_teeth",
                severity=WARNING,
                title=f"No adjacent tooth captured beside {tooth}",
                detail=(
                    f"Neither neighbour of tooth {tooth} was detected in the scan, "
                    "so proximal contacts cannot be established against real anatomy."
                ),
                recommendation=(
                    f"Extend the scan at least one tooth either side of {tooth}."
                ),
                tooth=tooth,
            )
        )
    return out


# ── 5. Opposing-arch sufficiency ──────────────────────────────────────────────

# Opposing arch below this share of the prepared arch's area is a stub scan.
_OPPOSING_AREA_RATIO = 0.45


def check_opposing_arch(
    prepared: trimesh.Trimesh, opposing: Optional[trimesh.Trimesh], prefix: str
) -> List[Finding]:
    if opposing is None:
        return [
            Finding(
                id=f"{prefix}-no-opposing",
                type="insufficient_opposing_arch",
                severity=CRITICAL,
                title="No opposing arch supplied",
                detail="Occlusion cannot be evaluated without the opposing arch.",
                recommendation="Request the opposing arch scan.",
            )
        ]
    ratio = float(opposing.area) / max(float(prepared.area), 1e-6)
    if ratio >= _OPPOSING_AREA_RATIO:
        return []
    return [
        Finding(
            id=f"{prefix}-opposing-small",
            type="insufficient_opposing_arch",
            severity=WARNING,
            title="Opposing arch coverage looks short",
            detail=(
                f"Opposing surface is {ratio*100:.0f}% of the prepared arch "
                f"({opposing.area:.0f} vs {prepared.area:.0f} mm²)."
            ),
            recommendation=(
                "Confirm the opposing scan spans the full occluding segment."
            ),
            measurements={"area_ratio": round(ratio, 3)},
        )
    ]


# ── 6. Bite-scan coverage ─────────────────────────────────────────────────────

# A buccal bite below this area carries too little overlap to register against.
_BITE_MIN_AREA_MM2 = 120.0
# Above this shell count the bite is shredded rather than a coherent patch.
_BITE_MAX_SHELLS = 40


def check_bite_coverage(
    bites: List[tuple], prefix: str
) -> List[Finding]:
    """bites: list of (label, trimesh)."""
    out: List[Finding] = []
    if not bites:
        return [
            Finding(
                id=f"{prefix}-no-bite",
                type="poor_bite_coverage",
                severity=CRITICAL,
                title="No bite registration supplied",
                detail="Upper and lower cannot be articulated without a bite scan.",
                recommendation="Request a buccal bite scan.",
            )
        ]

    for label, mesh in bites:
        shells = len(mesh.split(only_watertight=False))
        area = float(mesh.area)
        problems = []
        if area < _BITE_MIN_AREA_MM2:
            problems.append(f"only {area:.0f} mm² of surface")
        if shells > _BITE_MAX_SHELLS:
            problems.append(f"{shells} disconnected pieces")
        if not problems:
            continue
        out.append(
            Finding(
                id=f"{prefix}-bite-{label}",
                type="poor_bite_coverage",
                severity=CRITICAL,
                title=f"'{label}' is not usable for articulation",
                detail=(
                    f"{label} has " + " and ".join(problems) + ". A bite scan "
                    "registers the arches by overlapping both; fragmented or "
                    "sparse data gives an unreliable alignment."
                ),
                recommendation=(
                    "Request a fresh buccal bite covering an intact span of both "
                    "arches in occlusion."
                ),
                measurements={"area_mm2": round(area, 1), "shells": shells},
            )
        )
    return out


# ── 7. Per-tooth capture completeness ─────────────────────────────────────────

# A tooth region below this area is only partially captured. A molar crown is
# roughly 90-140 mm² of surface; a premolar 55-90.
_TOOTH_MIN_AREA_MM2 = 35.0
# Segmentation confidence under this means the region is a guess.
_LOW_CONFIDENCE = 0.55


def check_tooth_capture(seg: Segmentation, prefix: str) -> List[Finding]:
    out: List[Finding] = []
    for t in seg.teeth:
        if t.area_mm2 >= _TOOTH_MIN_AREA_MM2:
            continue
        name = f"tooth {t.fdi}" if t.fdi else f"arch position {t.arch_index}"
        out.append(
            Finding(
                id=f"{prefix}-partial-{t.label}",
                type="incomplete_tooth_capture",
                severity=WARNING,
                title=f"Partial capture at {name}",
                detail=(
                    f"Only {t.area_mm2:.0f} mm² of surface was segmented for {name}, "
                    "below the area of a fully captured crown."
                ),
                recommendation=f"Rescan {name} to capture the full crown.",
                tooth=t.fdi,
                measurements={"area_mm2": round(t.area_mm2, 1)},
            )
        )

    low = [t for t in seg.teeth if t.confidence < _LOW_CONFIDENCE]
    if low:
        out.append(
            Finding(
                id=f"{prefix}-low-confidence",
                type="segmentation_uncertain",
                severity=INFO,
                title=f"Segmentation uncertain on {len(low)} region(s)",
                detail=(
                    f"{len(low)} tooth region(s) scored under {_LOW_CONFIDENCE:.2f} "
                    "mean confidence. Prepared teeth are the usual cause — the "
                    "segmentation model is trained on natural dentition."
                ),
                recommendation=(
                    "Treat per-tooth attribution here as advisory; the mesh-level "
                    "checks are unaffected."
                ),
                measurements={"low_confidence_regions": len(low)},
            )
        )
    return out
