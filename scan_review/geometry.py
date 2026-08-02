"""Deterministic mesh inspection: holes, topology, and printability.

Everything in this file is measured, not inferred. No model is involved — given
the same STL it returns the same numbers every time, which is what makes the
output usable as a QA gate rather than a suggestion.

Two domain facts drive the whole design, and getting either wrong makes the
output worthless:

**1. STL has no vertex sharing.** A binary STL is a triangle *soup*: each
triangle carries its own three vertex positions, so the same corner is stored
once per touching face. Analysed as-is, every single edge looks like a boundary
edge and every mesh looks like it is 100% holes. Vertices must be merged by
position first. That is why the mesh is loaded with `process=True`, and why
`_merge_within_tolerance` exists — exporters that round coordinates leave
neighbouring triangles fractionally apart, which reads as thousands of hairline
"holes" that are not real. We report both the raw and the tolerance-merged
counts so that difference is visible instead of silently guessed at.

**2. An intraoral scan is not supposed to be watertight.** It is an open
surface patch — a shell of the arch — with one large boundary loop running
around its trimmed perimeter. Flagging "not watertight" on a raw scan would
fire on literally every case and train the lab to ignore the feature. So the
largest boundary loop is classified as the expected *trim boundary*, and only
the remaining loops are reported as actual holes. Callers that know better (a
designed crown, a die, anything headed for a printer) can pass
`expect_watertight=True` and then every open loop counts against the mesh.

Runs synchronously and is CPU-bound; callers must push it to a worker thread.
"""

import io
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from .config import settings

logger = logging.getLogger(__name__)

# Dental meshes are conventionally millimetres. STL carries no unit information
# at all, so this is an assumption — but it is the universal convention for
# intraoral scanners and CAD exports, and dimension checks below depend on it.
ASSUMED_UNITS = "mm"

# Plausible bounding-box extents for dental geometry, in mm. A full arch is
# ~45-75 mm wide; a single die is ~8-15 mm. Anything outside the outer envelope
# is almost always a unit-scale mistake (metres or centimetres exported as mm).
_MIN_PLAUSIBLE_EXTENT_MM = 3.0
_MAX_PLAUSIBLE_EXTENT_MM = 250.0

# A disconnected shell holding less than this share of total surface area is
# treated as debris (a stray blob of tongue, cheek, or scanner noise) rather
# than as a second real object.
_FRAGMENT_AREA_SHARE = 0.02

# Hole size tiers, in mm². Size matters more than count in dental work: twenty
# pinholes are a clean-up job, one 30 mm² void means the area was never
# captured and needs a rescan.
#   < 1 mm²   (~1.1 mm across) — minor; fillable without losing detail.
#   1–25 mm²                   — major; will not print or mill correctly.
#   >= 25 mm² (~5.6 mm across) — severe; a missing region, not a defect.
_MAJOR_HOLE_MM2 = 1.0
_SEVERE_HOLE_MM2 = 25.0

# A mesh with any CRITICAL condition is capped here regardless of the additive
# penalties, so the score can never read "minor issues" (70–89) while a finding
# says CRITICAL. The two are shown side by side; they must not contradict.
_CRITICAL_SCORE_CAP = 65.0

# Decimal places used by the tolerance re-merge. 4 digits = 0.1 µm at mm scale:
# far below any scanner's ~20 µm resolution, so it welds exporter float noise
# without collapsing genuine geometry.
_MERGE_DIGITS = 4


class MeshInspectError(RuntimeError):
    """Raised when a file cannot be parsed as a usable mesh."""


@dataclass
class Hole:
    """One closed boundary loop in the surface."""

    id: str
    edges: int
    perimeter_mm: float
    area_mm2: float
    max_diameter_mm: float
    centroid: List[float]
    # True for the loop we believe is the scan's outer trim edge, not a defect.
    is_trim_boundary: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "edges": self.edges,
            "perimeter_mm": round(self.perimeter_mm, 3),
            "area_mm2": round(self.area_mm2, 4),
            "max_diameter_mm": round(self.max_diameter_mm, 3),
            "centroid": [round(c, 3) for c in self.centroid],
            "is_trim_boundary": self.is_trim_boundary,
        }


@dataclass
class MeshReport:
    """Every measurement taken from one mesh file."""

    label: str
    file_type: str
    size_bytes: int

    vertices: int = 0
    faces: int = 0

    watertight: bool = False
    winding_consistent: bool = True
    euler_number: Optional[int] = None

    open_edges: int = 0
    open_edges_before_merge: int = 0
    non_manifold_edges: int = 0
    duplicate_faces: int = 0
    degenerate_faces: int = 0
    unreferenced_vertices: int = 0

    holes: List[Hole] = field(default_factory=list)
    trim_boundary: Optional[Hole] = None

    shells: int = 1
    fragments: List[Dict[str, Any]] = field(default_factory=list)

    surface_area_mm2: float = 0.0
    volume_mm3: Optional[float] = None
    bounding_box_mm: List[float] = field(default_factory=list)

    expect_watertight: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "units_assumed": ASSUMED_UNITS,
            "vertices": self.vertices,
            "faces": self.faces,
            "watertight": self.watertight,
            "winding_consistent": self.winding_consistent,
            "euler_number": self.euler_number,
            "open_edges": self.open_edges,
            "open_edges_before_merge": self.open_edges_before_merge,
            "non_manifold_edges": self.non_manifold_edges,
            "duplicate_faces": self.duplicate_faces,
            "degenerate_faces": self.degenerate_faces,
            "unreferenced_vertices": self.unreferenced_vertices,
            "hole_count": len(self.holes),
            "holes": [h.as_dict() for h in self.holes],
            "trim_boundary": self.trim_boundary.as_dict() if self.trim_boundary else None,
            "shells": self.shells,
            "fragments": self.fragments,
            "surface_area_mm2": round(self.surface_area_mm2, 2),
            "volume_mm3": round(self.volume_mm3, 2) if self.volume_mm3 is not None else None,
            "bounding_box_mm": [round(d, 2) for d in self.bounding_box_mm],
            "expect_watertight": self.expect_watertight,
        }


# ── Loading ───────────────────────────────────────────────────────────────────


def _load_mesh(data: bytes, file_type: str) -> trimesh.Trimesh:
    """Parse bytes into a single Trimesh, merging vertices by position.

    `process=True` is essential, not cosmetic — see the module docstring on STL
    triangle soup.
    """
    try:
        loaded = trimesh.load(
            file_obj=io.BytesIO(data),
            file_type=file_type,
            process=True,
            force="mesh",
        )
    except Exception as exc:  # noqa: BLE001 - any parser failure is a bad file
        raise MeshInspectError(f"Could not parse {file_type} file: {exc}") from exc

    # Some formats (obj/glb/3mf) can carry several parts. Concatenating gives one
    # surface to measure; the per-part split is recovered later as `shells`.
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise MeshInspectError("File contains no triangle geometry")
        loaded = trimesh.util.concatenate(geoms)

    if not isinstance(loaded, trimesh.Trimesh):
        raise MeshInspectError(
            f"File did not yield a triangle mesh (got {type(loaded).__name__})"
        )
    if len(loaded.faces) == 0:
        raise MeshInspectError("Mesh contains no faces")
    if len(loaded.faces) > settings.max_faces:
        raise MeshInspectError(
            f"Mesh has {len(loaded.faces)} faces, over the "
            f"{settings.max_faces} limit (SCAN_REVIEW_MAX_FACES)"
        )
    return loaded


def _merge_within_tolerance(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Re-merge vertices with a small positional tolerance.

    Exporters that write rounded coordinates leave adjacent triangles a few
    float-ULPs apart, which shows up as a huge number of one-edge "holes" that
    do not exist in the geometry. Welding at 0.1 µm removes that artefact
    without touching anything a scanner could actually resolve.
    """
    merged = mesh.copy()
    try:
        merged.merge_vertices(digits_vertex=_MERGE_DIGITS)
    except TypeError:
        # Older trimesh signatures took no keyword; the exact merge already ran
        # during load, so falling through is safe.
        merged.merge_vertices()
    return merged


# ── Topology ──────────────────────────────────────────────────────────────────


def _edge_counts(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    """Return (unique_edges, occurrence_count) over the mesh's sorted edges.

    An edge used once is a boundary; twice is manifold; three or more is a
    non-manifold junction where surfaces meet along a shared spine.
    """
    if len(mesh.faces) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return np.unique(mesh.edges_sorted, axis=0, return_counts=True)


def _trace_loops(boundary_edges: np.ndarray) -> List[List[int]]:
    """Chain boundary edges into closed loops.

    Walks by *edge* rather than by vertex so that pinch points — a vertex where
    two separate holes touch, which has four boundary neighbours instead of two
    — split into two loops instead of derailing the traversal.
    """
    if len(boundary_edges) == 0:
        return []

    adjacency: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for index, (a, b) in enumerate(boundary_edges):
        adjacency[int(a)].append((int(b), index))
        adjacency[int(b)].append((int(a), index))

    used: set = set()
    loops: List[List[int]] = []

    for start in list(adjacency.keys()):
        for first_neighbour, first_edge in adjacency[start]:
            if first_edge in used:
                continue
            used.add(first_edge)
            loop = [start]
            current = first_neighbour

            while current != start:
                loop.append(current)
                step = None
                for neighbour, edge_index in adjacency[current]:
                    if edge_index not in used:
                        step = (neighbour, edge_index)
                        break
                if step is None:
                    # Dangling chain rather than a closed loop. Real meshes
                    # produce these only from non-manifold debris; keep what we
                    # traced so the edge count is still reported.
                    break
                used.add(step[1])
                current = step[0]

            if len(loop) >= 3:
                loops.append(loop)

    return loops


def _loop_metrics(vertices: np.ndarray, loop: List[int], index: int) -> Hole:
    """Measure one closed loop: perimeter, projected area, size, position."""
    points = vertices[loop]
    closed = np.vstack([points, points[:1]])
    segments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(segments.sum())

    # Newell's method: the magnitude of the summed cross products is twice the
    # area of the polygon projected onto its own best-fit plane. Exact for a
    # planar loop and a good approximation for the gently curved loops that
    # bound a hole in a scan.
    cross = np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)
    area = float(np.linalg.norm(cross) * 0.5)

    centroid = points.mean(axis=0)
    # Max chord across the loop, sampled when the loop is large so that a
    # 50k-point trim boundary does not turn into a 2.5-billion-pair distance
    # matrix. The sample is a size estimate, not a precision measurement.
    sample = points if len(points) <= 512 else points[
        np.linspace(0, len(points) - 1, 512).astype(int)
    ]
    spread = float(
        np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1).max()
    ) if len(sample) > 1 else 0.0

    return Hole(
        id=f"h{index}",
        edges=len(loop),
        perimeter_mm=perimeter,
        area_mm2=area,
        max_diameter_mm=spread,
        centroid=[float(c) for c in centroid],
    )


def _classify_loops(
    loops: List[Hole], *, expect_watertight: Optional[bool]
) -> Tuple[List[Hole], Optional[Hole]]:
    """Split loops into real holes and the expected trim boundary.

    A raw intraoral scan legitimately has one large open perimeter. Unless the
    caller says the mesh should be closed, the longest loop is treated as that
    perimeter and excluded from the defect list.
    """
    if not loops:
        return [], None

    ordered = sorted(loops, key=lambda h: h.perimeter_mm, reverse=True)

    if expect_watertight:
        # Every opening is a defect in a mesh that should be closed.
        return ordered, None

    trim = ordered[0]
    trim.is_trim_boundary = True
    return ordered[1:], trim


# ── Inspection ────────────────────────────────────────────────────────────────


def inspect_mesh(
    data: bytes,
    file_type: str,
    *,
    label: str = "scan",
    expect_watertight: Optional[bool] = None,
) -> MeshReport:
    """Measure one mesh file. CPU-bound — run it in a worker thread.

    Raises MeshInspectError if the bytes are not a loadable mesh.
    """
    raw = _load_mesh(data, file_type)

    _unique_raw, counts_raw = _edge_counts(raw)
    open_before = int((counts_raw == 1).sum())

    # Weld exporter float noise before drawing any conclusion about holes.
    mesh = _merge_within_tolerance(raw) if open_before else raw

    unique_edges, counts = _edge_counts(mesh)
    boundary_edges = unique_edges[counts == 1]
    non_manifold = int((counts > 2).sum())

    report = MeshReport(
        label=label,
        file_type=file_type,
        size_bytes=len(data),
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        open_edges=int(len(boundary_edges)),
        open_edges_before_merge=open_before,
        non_manifold_edges=non_manifold,
        expect_watertight=expect_watertight,
    )

    # Topology flags. trimesh computes these lazily and a malformed mesh can
    # make any of them throw, so each is guarded independently — a failed
    # sub-check must not lose the measurements that did succeed.
    try:
        report.watertight = bool(mesh.is_watertight)
    except Exception:  # noqa: BLE001
        report.watertight = len(boundary_edges) == 0
    try:
        report.winding_consistent = bool(mesh.is_winding_consistent)
    except Exception:  # noqa: BLE001
        pass
    try:
        report.euler_number = int(mesh.euler_number)
    except Exception:  # noqa: BLE001
        pass

    # Duplicate faces: identical vertex triples, regardless of winding order.
    try:
        sorted_faces = np.sort(mesh.faces, axis=1)
        _uniq_faces, face_counts = np.unique(sorted_faces, axis=0, return_counts=True)
        report.duplicate_faces = int((face_counts - 1)[face_counts > 1].sum())
    except Exception:  # noqa: BLE001
        pass

    # Degenerate faces: a repeated vertex index, or effectively zero area.
    try:
        repeated = (
            (mesh.faces[:, 0] == mesh.faces[:, 1])
            | (mesh.faces[:, 1] == mesh.faces[:, 2])
            | (mesh.faces[:, 0] == mesh.faces[:, 2])
        )
        scale = float(mesh.scale) or 1.0
        slivers = mesh.area_faces < (scale * scale * 1e-10)
        report.degenerate_faces = int(np.count_nonzero(repeated | slivers))
    except Exception:  # noqa: BLE001
        pass

    try:
        referenced = np.unique(mesh.faces)
        report.unreferenced_vertices = int(len(mesh.vertices) - len(referenced))
    except Exception:  # noqa: BLE001
        pass

    # Boundary loops → holes.
    loops = _trace_loops(boundary_edges)
    measured = [
        _loop_metrics(mesh.vertices, loop, i + 1) for i, loop in enumerate(loops)
    ]
    report.holes, report.trim_boundary = _classify_loops(
        measured, expect_watertight=expect_watertight
    )
    # Largest first: the lab should see the worst hole at the top of the list.
    report.holes.sort(key=lambda h: h.area_mm2, reverse=True)

    # Disconnected shells. Anything tiny is scanner debris, not a second object.
    try:
        components = trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
        )
        report.shells = int(len(components))
        total_area = float(mesh.area) or 1.0
        for comp in components:
            comp_area = float(mesh.area_faces[comp].sum())
            if comp_area / total_area < _FRAGMENT_AREA_SHARE:
                report.fragments.append(
                    {
                        "faces": int(len(comp)),
                        "area_mm2": round(comp_area, 3),
                        "area_share": round(comp_area / total_area, 5),
                    }
                )
    except Exception:  # noqa: BLE001
        pass

    try:
        report.surface_area_mm2 = float(mesh.area)
    except Exception:  # noqa: BLE001
        pass
    try:
        report.bounding_box_mm = [float(e) for e in mesh.extents]
    except Exception:  # noqa: BLE001
        pass
    # Volume is only meaningful for a closed mesh; on an open surface trimesh
    # still returns a number and it is nonsense, so it is withheld.
    if report.watertight:
        try:
            report.volume_mm3 = float(abs(mesh.volume))
        except Exception:  # noqa: BLE001
            pass

    return report


# ── Findings ──────────────────────────────────────────────────────────────────


def _finding(
    fid: str, category: str, severity: str, title: str, detail: str, recommendation: str
) -> Dict[str, Any]:
    return {
        "id": fid,
        "category": category,
        "severity": severity,
        "title": title,
        "detail": detail,
        "recommendation": recommendation,
        # Marks this as measured geometry rather than model commentary, so the
        # UI can present it with more confidence than an LLM observation.
        "source": "geometry",
    }


def build_findings(report: MeshReport) -> List[Dict[str, Any]]:
    """Turn measurements into categorised, actionable findings.

    Categories match the vocabulary the existing case-review UI already renders
    (MESH_QUALITY / COMPLETENESS / PRINT_READINESS / DIMENSIONS), so this output
    can be displayed without a new component.
    """
    findings: List[Dict[str, Any]] = []
    prefix = report.label.replace(" ", "_").lower() or "scan"
    n = 0

    def next_id() -> str:
        nonlocal n
        n += 1
        return f"{prefix}-g{n}"

    # Holes — the headline check.
    if report.holes:
        biggest = report.holes[0]
        severe = [h for h in report.holes if h.area_mm2 >= _SEVERE_HOLE_MM2]
        major = [h for h in report.holes if h.area_mm2 >= _MAJOR_HOLE_MM2]
        severity = "CRITICAL" if major else "WARNING"

        if severe:
            verdict = (
                f"{len(severe)} exceed {_SEVERE_HOLE_MM2:.0f} mm² — that is a "
                "region the scanner never captured, not a defect to patch."
            )
            action = (
                "Rescan the affected area. Filling a void this size invents "
                "surface that was never measured, which is unsafe on a margin "
                "or occlusal contact."
            )
        elif major:
            verdict = (
                f"{len(major)} exceed {_MAJOR_HOLE_MM2:.0f} mm² and will not "
                "print or mill cleanly."
            )
            action = (
                "Fill the holes before design or printing, and rescan instead "
                "if any sits on a margin or occlusal surface."
            )
        else:
            verdict = (
                f"All are under {_MAJOR_HOLE_MM2:.0f} mm² — small, but they "
                "still break a solid model."
            )
            action = "Run a hole-fill during clean-up before design."

        findings.append(
            _finding(
                next_id(),
                "PRINT_READINESS",
                severity,
                f"{len(report.holes)} hole(s) in the surface",
                (
                    f"Found {len(report.holes)} closed boundary loop(s) inside the "
                    f"mesh. The largest spans {biggest.area_mm2:.2f} mm² "
                    f"({biggest.max_diameter_mm:.2f} mm across) at "
                    f"[{', '.join(f'{c:.1f}' for c in biggest.centroid)}]. "
                    + verdict
                ),
                action,
            )
        )

    if report.expect_watertight and not report.watertight:
        findings.append(
            _finding(
                next_id(),
                "PRINT_READINESS",
                "CRITICAL",
                "Mesh is not watertight",
                (
                    f"{report.open_edges} open edge(s) remain. This file was "
                    "expected to be a closed solid."
                ),
                "Close the mesh before sending it to a printer or mill.",
            )
        )

    if report.non_manifold_edges:
        findings.append(
            _finding(
                next_id(),
                "MESH_QUALITY",
                "CRITICAL" if report.non_manifold_edges > 50 else "WARNING",
                f"{report.non_manifold_edges} non-manifold edge(s)",
                (
                    f"{report.non_manifold_edges} edge(s) are shared by three or "
                    "more faces. Most slicers and CAM packages either reject this "
                    "or silently produce wrong geometry."
                ),
                "Repair the non-manifold junctions; do not send this file to production as-is.",
            )
        )

    if not report.winding_consistent:
        findings.append(
            _finding(
                next_id(),
                "MESH_QUALITY",
                "WARNING",
                "Inconsistent face winding",
                (
                    "Face normals do not agree on which side is outside, so "
                    "inside and outside are ambiguous."
                ),
                "Recalculate/unify normals before design work.",
            )
        )

    if report.fragments:
        total_fragment_faces = sum(f["faces"] for f in report.fragments)
        findings.append(
            _finding(
                next_id(),
                "MESH_QUALITY",
                "WARNING",
                f"{len(report.fragments)} floating fragment(s)",
                (
                    f"{len(report.fragments)} disconnected piece(s) totalling "
                    f"{total_fragment_faces} face(s) sit apart from the main "
                    "surface — typically captured tongue, cheek, saliva, or "
                    "scanner noise."
                ),
                "Delete the stray fragments so they do not end up in the design or on the model.",
            )
        )

    if report.degenerate_faces:
        findings.append(
            _finding(
                next_id(),
                "MESH_QUALITY",
                "INFO" if report.degenerate_faces < 20 else "WARNING",
                f"{report.degenerate_faces} degenerate face(s)",
                (
                    f"{report.degenerate_faces} face(s) have zero area or a "
                    "repeated vertex. They carry no geometry and can trip up "
                    "downstream tools."
                ),
                "Run a mesh clean-up to drop degenerate triangles.",
            )
        )

    if report.duplicate_faces:
        findings.append(
            _finding(
                next_id(),
                "MESH_QUALITY",
                "INFO",
                f"{report.duplicate_faces} duplicate face(s)",
                f"{report.duplicate_faces} face(s) are exact repeats of another face.",
                "Remove duplicate faces during clean-up.",
            )
        )

    # Dimensions — almost always a unit-scale error when it fires.
    if report.bounding_box_mm:
        largest = max(report.bounding_box_mm)
        if largest < _MIN_PLAUSIBLE_EXTENT_MM or largest > _MAX_PLAUSIBLE_EXTENT_MM:
            findings.append(
                _finding(
                    next_id(),
                    "DIMENSIONS",
                    "CRITICAL",
                    "Implausible scale for dental geometry",
                    (
                        "Bounding box is "
                        f"{' × '.join(f'{d:.1f}' for d in report.bounding_box_mm)} mm. "
                        "Dental scans assume millimetres; this is far outside the "
                        "range of an arch or a die, which nearly always means the "
                        "file was exported in the wrong units."
                    ),
                    "Confirm the export units (mm) and re-export before using this file.",
                )
            )

    # Coverage — a scan too small to be the arch it claims to be.
    if report.surface_area_mm2 and report.surface_area_mm2 < 100.0:
        findings.append(
            _finding(
                next_id(),
                "COMPLETENESS",
                "WARNING",
                "Very small captured surface",
                (
                    f"Total surface area is only {report.surface_area_mm2:.1f} mm², "
                    "which is well below a normal arch or quadrant capture."
                ),
                "Check that the full prepared area and adjacent teeth were captured.",
            )
        )

    if not findings:
        findings.append(
            _finding(
                next_id(),
                "MESH_QUALITY",
                "INFO",
                "No geometry defects detected",
                (
                    f"{report.faces} faces, {report.shells} shell(s), no interior "
                    "holes, no non-manifold edges, consistent winding."
                ),
                "No geometry remediation needed.",
            )
        )

    return findings


def has_critical_defect(report: MeshReport) -> bool:
    """Whether the mesh has a condition that blocks production.

    Single source of truth for "this is CRITICAL", used by both the findings and
    the score so the two can never disagree.
    """
    if any(h.area_mm2 >= _MAJOR_HOLE_MM2 for h in report.holes):
        return True
    if report.expect_watertight and not report.watertight:
        return True
    if report.non_manifold_edges > 50:
        return True
    if report.bounding_box_mm:
        largest = max(report.bounding_box_mm)
        if largest < _MIN_PLAUSIBLE_EXTENT_MM or largest > _MAX_PLAUSIBLE_EXTENT_MM:
            return True
    return False


def score_report(report: MeshReport) -> float:
    """Reproducible 0–100 quality score from the measurements.

    Penalties are additive and capped per category so one bad dimension cannot
    alone zero the score. Any CRITICAL condition then caps the result, keeping
    the number consistent with the severities shown next to it.

    This is the value the endpoint returns as `overall_score` — deliberately
    computed here rather than asked of a model, so two runs over the same STL
    always agree.
    """
    score = 100.0

    # Holes: weighted by how big they are, not merely how many there are.
    if report.holes:
        severe = sum(1 for h in report.holes if h.area_mm2 >= _SEVERE_HOLE_MM2)
        major = sum(
            1 for h in report.holes if _MAJOR_HOLE_MM2 <= h.area_mm2 < _SEVERE_HOLE_MM2
        )
        minor = len(report.holes) - severe - major
        score -= min(60.0, severe * 30.0 + major * 12.0 + minor * 4.0)

    if report.expect_watertight and not report.watertight:
        score -= 15.0

    if report.non_manifold_edges:
        score -= min(20.0, 8.0 + math.log10(report.non_manifold_edges + 1) * 8.0)

    if not report.winding_consistent:
        score -= 10.0

    if report.fragments:
        score -= min(15.0, len(report.fragments) * 4.0)

    if report.degenerate_faces:
        score -= min(8.0, 2.0 + math.log10(report.degenerate_faces + 1) * 3.0)

    if report.duplicate_faces:
        score -= min(5.0, 1.0 + math.log10(report.duplicate_faces + 1) * 2.0)

    if report.bounding_box_mm:
        largest = max(report.bounding_box_mm)
        if largest < _MIN_PLAUSIBLE_EXTENT_MM or largest > _MAX_PLAUSIBLE_EXTENT_MM:
            score -= 25.0

    if has_critical_defect(report):
        score = min(score, _CRITICAL_SCORE_CAP)

    return round(max(0.0, min(100.0, score)), 1)


def risk_from_findings(findings: List[Dict[str, Any]], score: float) -> str:
    """Map findings + score onto the LOW/MEDIUM/HIGH vocabulary already in use."""
    if any(f.get("severity") == "CRITICAL" for f in findings):
        return "HIGH"
    if score < 60:
        return "HIGH"
    if score < 85 or any(f.get("severity") == "WARNING" for f in findings):
        return "MEDIUM"
    return "LOW"
