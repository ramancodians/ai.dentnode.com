"""HTTP surface for the Scan Review module.

Self-contained FastAPI router so the module can be mounted into this service or
lifted into its own deployment without touching its internals — `server.py` only
does `app.include_router(scan_review_router)`.

Route naming: the exact path `POST /scan-review` is already taken by Laby's
vision review of rendered arch images. This module owns the `/scan-review/*`
sub-namespace instead, so the two coexist without breaking Node's existing
integration.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from agent.config import settings as service_settings
from agent.usage import report_usage

from .config import settings
from .review import ScanReviewError, generate_scan_review

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scan-review", tags=["scan-review"])

# Metering feature key. Distinct from Laby's "scan_review" so the two show up
# separately in the AiUsageEvent ledger and cost can be attributed correctly.
FEATURE = "scan_review_mesh"


class ScanFile(BaseModel):
    url: str = Field(..., min_length=1, description="Direct URL to an STL/PLY/OBJ mesh")
    label: str = Field(default="scan", description='e.g. "Upper arch"')
    # None = auto (treat the largest boundary loop as the scan's trim edge).
    # True for anything that should be a closed solid: a die, a designed crown,
    # a print-ready model.
    expect_watertight: Optional[bool] = Field(default=None)


class ScanReviewRequest(BaseModel):
    lab_id: str = Field(..., min_length=1)
    files: List[ScanFile] = Field(..., min_length=1)
    case_context: Dict[str, Any] = Field(default_factory=dict)
    # Caller owns "today" so the date matches the rest of the case review.
    today: Optional[str] = None
    user_id: Optional[str] = None
    # Per-request override of SCAN_REVIEW_ENABLE_LLM — geometry-only runs are
    # free and deterministic, which is what you want for bulk backfills.
    use_llm: Optional[bool] = None


def _require_internal_key(provided: Optional[str]) -> None:
    if not service_settings.internal_key:
        raise HTTPException(status_code=500, detail="INTERNAL_API_KEY not configured")
    if not provided or provided != service_settings.internal_key:
        raise HTTPException(status_code=401, detail="Invalid internal key")


@router.get("/health")
async def scan_review_health() -> Dict[str, Any]:
    """Module-level readiness: reports config without exposing secrets."""
    return {
        "status": "healthy",
        "module": "scan-review",
        "model": settings.model,
        "llm_enabled": settings.enable_llm,
        "max_files": settings.max_files,
        "max_bytes": settings.max_bytes,
        "host_allowlist": sorted(settings.allowed_hosts) or None,
        "insecure_fetch": settings.allow_insecure_fetch,
    }


@router.post("/analyze")
async def analyze(
    body: ScanReviewRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
) -> Dict[str, Any]:
    """Fetch every mesh URL, measure it, and return a combined QA review."""
    _require_internal_key(x_internal_key or x_internal_id)

    today = body.today or datetime.now(timezone.utc).date().isoformat()

    try:
        result = await generate_scan_review(
            files=[f.model_dump() for f in body.files],
            case_context=body.case_context,
            today=today,
            use_llm=body.use_llm,
        )
    except ScanReviewError as exc:
        # Request-level rejection (no files / too many). No model was called, so
        # there is nothing to meter.
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scan review failed", extra={"lab_id": body.lab_id})
        raise HTTPException(status_code=500, detail=f"Scan review failed: {exc}")

    # Meter only when the model actually ran. A geometry-only review costs
    # nothing and must not create a billable event.
    if result.model is not None or result.llm_error is not None:
        _report(body, result)

    return {
        "success": True,
        "summary": result.summary,
        "risk_level": result.risk_level,
        "overall_score": result.overall_score,
        "findings": result.findings,
        "flags": result.flags,
        "files": [f.as_dict() for f in result.files],
        "analyzed_count": result.analyzed_count,
        "requested_count": len(result.files),
        "model": result.model,
        "parsed": result.parsed,
        "llm_error": result.llm_error,
        "raw_response": result.raw_response,
    }


def _report(body: ScanReviewRequest, result) -> None:
    """Fire-and-forget usage metering.

    Imported from `server` at call time rather than at module import so the
    router stays independently importable (and testable) without pulling the
    whole app in.
    """
    from server import _fire_and_forget

    _fire_and_forget(
        report_usage(
            feature=FEATURE,
            lab_id=body.lab_id,
            user_id=body.user_id,
            model=result.model or settings.model,
            usage=result.usage,
            cost=result.cost_usd,
            cost_source="openrouter" if result.cost_usd is not None else "estimated",
            latency_ms=result.latency_ms,
            status="error" if result.llm_error else "ok",
            meta={
                "files": len(result.files),
                "analyzed": result.analyzed_count,
                "score": result.overall_score,
                "risk": result.risk_level,
                "parsed": result.parsed,
            },
        )
    )
