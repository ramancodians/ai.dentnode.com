"""HTTP surface for Scan QA.

Owns `/scan-qa/*`, kept separate from both `/scan-review` (Laby's vision review)
and `/scan-review/analyze` (the mesh-measurement module). Scan QA is the
segmentation-conditioned pipeline; it reuses the SSRF-guarded fetcher from
`scan_review.fetch` rather than dereferencing caller URLs itself.

The response is shaped for a 3-D viewer: every finding that has a position
carries `location` / `locations` in the scan's own millimetre coordinates, which
is the same space the DN3D viewer renders the STL in (it applies no centering
transform), so the client can use them as world positions unchanged.
"""

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from agent.config import settings as service_settings
from agent.usage import report_usage
from scan_review.config import settings as fetch_settings
from scan_review.fetch import fetch_all

from .pipeline import ScanRole, run_scan_qa

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scan-qa", tags=["scan-qa"])

FEATURE = "scan_qa"

# Roles the pipeline knows how to weight. Anything else is scored with the
# default weight rather than rejected — an unexpected role should degrade, not
# fail the whole case.
KNOWN_ROLES = {"prepared_arch", "opposing_arch", "bite"}


class ScanQAFile(BaseModel):
    url: str = Field(..., min_length=1, description="Direct URL to an STL/PLY/OBJ mesh")
    label: str = Field(default="scan")
    role: str = Field(
        default="prepared_arch",
        description='"prepared_arch" | "opposing_arch" | "bite"',
    )
    jaw: str = Field(default="upper", description='"upper" | "lower"')


class ScanQARequest(BaseModel):
    lab_id: str = Field(..., min_length=1)
    files: List[ScanQAFile] = Field(..., min_length=1)
    case_context: Dict[str, Any] = Field(default_factory=dict)
    # FDI numbers of the prepared teeth, from the case prescription.
    prep_teeth: List[int] = Field(default_factory=list)
    # Segmentation is the slow part (~10s/arch on CPU) and its per-tooth output
    # is unreliable on prepared arches, so callers that only need the mesh-level
    # findings can skip it.
    segment: bool = True
    narrate: bool = True
    user_id: Optional[str] = None


def _require_internal_key(provided: Optional[str]) -> None:
    if not service_settings.internal_key:
        raise HTTPException(status_code=500, detail="INTERNAL_API_KEY not configured")
    if not provided or provided != service_settings.internal_key:
        raise HTTPException(status_code=401, detail="Invalid internal key")


@router.get("/health")
async def scan_qa_health() -> Dict[str, Any]:
    from .segmentation import _VENDOR

    weights = {
        j: (Path(_VENDOR) / f"MeshSegNet_{j}.zip").exists() for j in ("Max", "Man")
    }
    return {
        "status": "healthy",
        "module": "scan-qa",
        "segmentation_backend": "meshsegnet",
        "weights_present": weights,
        "host_allowlist": sorted(fetch_settings.allowed_hosts) or None,
    }


@router.post("/analyze")
async def analyze(
    body: ScanQARequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
) -> Dict[str, Any]:
    """Fetch every mesh, run the QA pipeline, return findings with locations."""
    _require_internal_key(x_internal_key or x_internal_id)

    specs = [
        {"url": f.url, "label": f.label, "expect_watertight": None} for f in body.files
    ]
    fetched = await fetch_all(specs)

    # The pipeline reads from disk, so the fetched bytes are staged in a temp dir
    # that is removed on the way out.
    with tempfile.TemporaryDirectory(prefix="scanqa-") as tmp:
        roles: List[ScanRole] = []
        fetch_errors: List[Dict[str, str]] = []

        for spec, mesh, err in fetched:
            src = next(f for f in body.files if f.url == spec["url"])
            if err is not None or mesh is None:
                fetch_errors.append({"label": src.label, "error": err or "not fetched"})
                continue
            suffix = f".{(mesh.file_type or 'stl').lstrip('.')}"
            path = Path(tmp) / f"{len(roles)}_{src.label.replace(' ', '_')}{suffix}"
            path.write_bytes(mesh.data)
            roles.append(
                ScanRole(
                    label=src.label,
                    path_or_mesh=path,
                    role=src.role if src.role in KNOWN_ROLES else "prepared_arch",
                    jaw=src.jaw,
                )
            )

        if not roles:
            raise HTTPException(
                status_code=400,
                detail=f"No mesh could be fetched. {fetch_errors}",
            )

        segmenter = None
        if body.segment:
            try:
                from .segmentation import MeshSegNetSegmenter

                segmenter = MeshSegNetSegmenter()
            except Exception as exc:  # noqa: BLE001
                # Missing weights or torch must not lose the geometric findings.
                logger.warning("Segmenter unavailable", extra={"error": str(exc)})

        try:
            report = run_scan_qa(
                roles, segmenter=segmenter, prep_teeth=body.prep_teeth
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Scan QA failed", extra={"lab_id": body.lab_id})
            raise HTTPException(status_code=500, detail=f"Scan QA failed: {exc}")

    if body.narrate:
        from .narrate import narrate

        report = await narrate(report, case_context=body.case_context)
        _report_usage(body, report)

    out = report.as_dict()
    out["success"] = True
    out["fetch_errors"] = fetch_errors
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    return out


def _report_usage(body: ScanQARequest, report) -> None:
    """Meter the narration call only — the geometry and segmentation are local
    CPU work with no per-call vendor cost."""
    from server import _fire_and_forget

    _fire_and_forget(
        report_usage(
            feature=FEATURE,
            lab_id=body.lab_id,
            user_id=body.user_id,
            model=service_settings.scan_review_vision_model,
            usage={},
            cost=None,
            cost_source="estimated",
            latency_ms=report.elapsed_ms,
            status="ok" if report.summary else "error",
            meta={
                "files": len(report.files),
                "score": report.overall_score,
                "risk": report.risk_level,
                "findings": len(report.findings),
            },
        )
    )
