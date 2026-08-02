"""FastAPI entrypoint for the Laby ADK agent service.

Endpoints:
  GET  /health         — liveness/readiness (no auth).
  POST /agent/run      — run one turn; streams normalized NDJSON events.

This service is internal-only. The Node backend calls /agent/run with the
shared x-internal-key or x-internal-id. It is deployed to Cloud Run with --ingress all.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.case_from_image import CaseExtractionError, extract_case_from_image
from agent.config import settings
from agent.insights import generate_insights
from agent.image_to_entry import ImageToEntryResult, image_to_entry
from agent.logging_setup import setup_logging
from agent.marketing_copy import generate_product_update_email
from agent.openrouter import OpenRouterError
from agent.rejected_cases import (
    RejectedCasesParseError,
    generate_rejected_cases_report,
)
from agent.rx_review import generate_rx_review
from agent.runner import run_turn
from agent.browser_runner import run_browser_turn
from agent.scan_review import generate_scan_review
from agent.usage import report_usage

# Standalone module — not part of Laby. Owns the /scan-review/* sub-namespace
# (mesh QA from raw STL URLs); the flat POST /scan-review below is Laby's
# separate vision review over rendered arch images.
from scan_review import scan_review_router
from scan_review.config import settings as scan_review_settings

# Segmentation-conditioned scan QA. Owns /scan-qa/*; findings carry mesh
# coordinates so the DN3D viewer can pin them to the model.
from scan_qa import scan_qa_router

# Wire JSON logging before the first log line is emitted.
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.validate()
    scan_review_settings.validate()
    logger.info("Laby agent started", extra={"model": settings.model, "port": settings.port})
    if scan_review_settings.allow_insecure_fetch:
        # This disables the public-IP requirement on outbound mesh fetches. It
        # is a local-dev switch; seeing it in a deployed log is an incident.
        logger.warning(
            "Scan Review insecure fetch is ENABLED — private/loopback mesh URLs "
            "are reachable. Never use this outside local development."
        )
    yield


app = FastAPI(title="Laby ADK Agent", version="1.0.0", lifespan=lifespan)

app.include_router(scan_review_router)
app.include_router(scan_qa_router)


class HistoryTurn(BaseModel):
    role: str
    text: str


class MentionItem(BaseModel):
    id: str
    name: str
    role: str


class RunRequest(BaseModel):
    lab_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    history: Optional[List[HistoryTurn]] = None
    mentions: Optional[List[MentionItem]] = None


class InsightsRequest(BaseModel):
    lab_id: str = Field(..., min_length=1)
    data: Dict[str, Any] = Field(default_factory=dict)
    question: Optional[str] = None
    user_id: Optional[str] = None


class RxReviewRequest(BaseModel):
    # Per-agent user inputs are built by Node (buildCaseInput) and passed verbatim.
    lab_id: str = Field(..., min_length=1)
    completeness: str = Field(...)
    clinical: str = Field(...)
    instructions: str = Field(...)
    user_id: Optional[str] = None


class RejectedCaseImage(BaseModel):
    image_base64: str = Field(..., min_length=1)
    mime_type: Optional[str] = None


class RejectedCasesRequest(BaseModel):
    lab_id: str = Field(..., min_length=1)
    lab_name: str = Field(default="")
    cases: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[RejectedCaseImage] = Field(default_factory=list)


class ProductUpdateRequest(BaseModel):
    """The one AI call with no owning lab (see PLATFORM_LAB_ID)."""

    commits: List[Dict[str, Any]] = Field(default_factory=list)


class ScanViewItem(BaseModel):
    """One rendered arch view. Node renders these from the STL (stlPreviews.ts)."""

    label: str = Field(default="view")
    mime_type: Optional[str] = None
    image_base64: str = Field(..., min_length=1)


class ScanReviewRequest(BaseModel):
    lab_id: str = Field(..., min_length=1)
    case_context: Dict[str, Any] = Field(default_factory=dict)
    # May be empty: the prompt tells the model to fall back to case context only,
    # matching the old Node behaviour when no preview could be rendered.
    views: List[ScanViewItem] = Field(default_factory=list)
    preview_errors: List[str] = Field(default_factory=list)
    # Node owns "today" so the date matches the rest of the case review.
    today: Optional[str] = None
    user_id: Optional[str] = None


class CaseFromImageRequest(BaseModel):
    # Node fetches the image (allowlist/SSRF-checked) and sends the bytes; this
    # service never dereferences a URL.
    lab_id: str = Field(..., min_length=1)
    image_base64: str = Field(..., min_length=1)
    mime_type: Optional[str] = None
    user_id: Optional[str] = None


class ImageToEntryRequest(BaseModel):
    """Image-only extraction; lab/user fields exist solely for auth metering."""

    lab_id: str = Field(..., min_length=1)
    image_base64: str = Field(..., min_length=1)
    mime_type: Optional[str] = None
    user_id: Optional[str] = None


# Every AiUsageEvent needs a lab_id, but the weekly product-update email is
# platform marketing — it belongs to no lab. Rather than drop the row (and lose
# the spend from the ledger) or invent a fake lab, it is attributed to this
# reserved sentinel. It can never collide with a real id: Lab.id values are
# cuids. Any per-lab billing/quota query must exclude it.
PLATFORM_LAB_ID = "__platform__"


# Holds references to fire-and-forget metering tasks so they are not garbage
# collected before they finish. Metering must never delay the reply, so it runs
# out-of-band via asyncio.create_task rather than being awaited inline.
_background_tasks: set = set()


def _fire_and_forget(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _require_internal_key(provided: Optional[str]) -> None:
    if not settings.internal_key:
        raise HTTPException(status_code=500, detail="INTERNAL_API_KEY not configured")
    if not provided or provided != settings.internal_key:
        raise HTTPException(status_code=401, detail="Invalid internal key")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "laby-adk",
        "model": settings.model,
        "provider": "openrouter",
    }


@app.post("/browser/run")
async def browser_run(
    body: RunRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
) -> StreamingResponse:
    """Isolated DNLink browser-controller endpoint; it never loads Laby."""
    _require_internal_key(x_internal_key or x_internal_id)
    history = [t.model_dump() for t in (body.history or [])]
    async def event_stream():
        async for event in run_browser_turn(lab_id=body.lab_id, user_id=body.user_id, question=body.question, history=history):
            yield json.dumps(event, ensure_ascii=False) + "\n"
    return StreamingResponse(event_stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache, no-transform"})

@app.post("/agent/run")
async def agent_run(
    body: RunRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
) -> StreamingResponse:
    provided_key = x_internal_key or x_internal_id
    _require_internal_key(provided_key)

    history = [t.model_dump() for t in (body.history or [])]
    mentions = [m.model_dump() for m in (body.mentions or [])]
    t0 = time.monotonic()

    async def event_stream():
        try:
            async for event in run_turn(
                lab_id=body.lab_id,
                user_id=body.user_id,
                question=body.question,
                history=history,
                mentions=mentions,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        finally:
            elapsed = time.monotonic() - t0
            logger.info(
                "Agent turn stream finished",
                extra={"lab_id": body.lab_id, "elapsed_secs": round(elapsed, 2)},
            )

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache, no-transform"},
    )


@app.post("/insights")
async def insights(
    body: InsightsRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
) -> Dict[str, Any]:
    provided_key = x_internal_key or x_internal_id
    _require_internal_key(provided_key)

    try:
        result = await generate_insights(data=body.data, question=body.question)
    except OpenRouterError as exc:
        logger.error(
            "Insights generation failed",
            extra={"lab_id": body.lab_id, "error": str(exc)},
        )
        # Best-effort error metering (never awaited on the hot path).
        _fire_and_forget(
            report_usage(
                feature="insights",
                lab_id=body.lab_id,
                user_id=body.user_id,
                model=settings.model,
                status="error",
            )
        )
        raise HTTPException(status_code=502, detail="Insights model call failed")

    # Meter the successful call out-of-band with OpenRouter's exact cost so the
    # reply is never delayed or broken by metering.
    _fire_and_forget(
        report_usage(
            feature="insights",
            lab_id=body.lab_id,
            user_id=body.user_id,
            model=settings.model,
            usage=result.usage,
            cost=result.cost_usd,
            cost_source="openrouter",
            latency_ms=result.latency_ms,
            status="ok",
        )
    )

    return {"success": True, "insights": result.text}


@app.post("/rx-review")
async def rx_review(
    body: RxReviewRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
) -> Dict[str, Any]:
    provided_key = x_internal_key or x_internal_id
    _require_internal_key(provided_key)

    try:
        result = await generate_rx_review(
            completeness=body.completeness,
            clinical=body.clinical,
            instructions=body.instructions,
        )
    except Exception as exc:  # noqa: BLE001 - unexpected failure, meter + surface
        logger.error(
            "RX review generation failed",
            extra={"lab_id": body.lab_id, "error": str(exc)},
        )
        _fire_and_forget(
            report_usage(
                feature="rx_review",
                lab_id=body.lab_id,
                user_id=body.user_id,
                model=settings.model,
                status="error",
            )
        )
        raise HTTPException(status_code=502, detail="RX review model call failed")

    # Meter out-of-band. Each sub-agent degrades to empty findings on its own
    # failure (mirroring Node); if every sub-agent failed we still return a valid
    # (empty) result so Node completes the review, but meter status="error".
    _fire_and_forget(
        report_usage(
            feature="rx_review",
            lab_id=body.lab_id,
            user_id=body.user_id,
            model=result.model,
            usage=result.usage,
            cost=result.cost_usd,
            cost_source="openrouter" if result.cost_usd is not None else "estimated",
            latency_ms=result.latency_ms,
            status="ok" if result.any_ok else "error",
        )
    )

    return {
        "success": True,
        "completeness": result.completeness,
        "clinical": result.clinical,
        "instructions": result.instructions,
        "model": result.model,
    }


@app.post("/scan-review")
async def scan_review(
    body: ScanReviewRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
) -> Dict[str, Any]:
    provided_key = x_internal_key or x_internal_id
    _require_internal_key(provided_key)

    today = body.today or datetime.now(timezone.utc).date().isoformat()

    try:
        result = await generate_scan_review(
            case_context=body.case_context,
            views=[v.model_dump() for v in body.views],
            preview_errors=body.preview_errors,
            today=today,
        )
    except OpenRouterError as exc:
        logger.error(
            "Scan review generation failed",
            extra={"lab_id": body.lab_id, "error": str(exc)},
        )
        _fire_and_forget(
            report_usage(
                feature="scan_review",
                lab_id=body.lab_id,
                user_id=body.user_id,
                model=settings.scan_review_vision_model,
                status="error",
            )
        )
        raise HTTPException(status_code=502, detail="Scan review model call failed")

    # An unparseable reply still spent tokens and still returns raw text to Node
    # (which falls back to it as the summary), so meter it as ok but flag the
    # parse failure for later cost/quality analysis.
    _fire_and_forget(
        report_usage(
            feature="scan_review",
            lab_id=body.lab_id,
            user_id=body.user_id,
            model=result.model,
            usage=result.usage,
            cost=result.cost_usd,
            cost_source="openrouter" if result.cost_usd is not None else "estimated",
            latency_ms=result.latency_ms,
            status="ok",
            meta={"views": len(body.views), "parsed": result.parsed},
        )
    )

    return {
        "success": True,
        "summary": result.summary,
        "risk_level": result.risk_level,
        "overall_score": result.overall_score,
        "findings": result.findings,
        "flags": result.flags,
        "raw_response": result.raw_response,
        "model": result.model,
        "parsed": result.parsed,
    }


@app.post("/case-from-image")
async def case_from_image(
    body: CaseFromImageRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
) -> Dict[str, Any]:
    provided_key = x_internal_key or x_internal_id
    _require_internal_key(provided_key)

    def _meter_error() -> None:
        _fire_and_forget(
            report_usage(
                feature="case_from_image",
                lab_id=body.lab_id,
                user_id=body.user_id,
                model=settings.vision_model,
                status="error",
            )
        )

    try:
        result = await extract_case_from_image(
            image_base64=body.image_base64,
            mime_type=body.mime_type or "image/jpeg",
        )
    except CaseExtractionError as exc:
        # Model replied, but not with usable JSON. Tokens were still spent, so
        # this is metered as an error rather than dropped.
        logger.error(
            "Case extraction returned unparseable output",
            extra={"lab_id": body.lab_id, "error": str(exc)},
        )
        _meter_error()
        raise HTTPException(status_code=502, detail=str(exc))
    except OpenRouterError as exc:
        logger.error(
            "Case extraction model call failed",
            extra={"lab_id": body.lab_id, "error": str(exc)},
        )
        _meter_error()
        raise HTTPException(status_code=502, detail="Case extraction model call failed")

    _fire_and_forget(
        report_usage(
            feature="case_from_image",
            lab_id=body.lab_id,
            user_id=body.user_id,
            model=result.model,
            usage=result.usage,
            cost=result.cost_usd,
            cost_source="openrouter" if result.cost_usd is not None else "estimated",
            latency_ms=result.latency_ms,
            status="ok",
        )
    )

    return {"success": True, "extracted": result.extracted, "model": result.model}


@app.post("/image-to-entry")
async def image_to_entry_endpoint(
    body: ImageToEntryRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
) -> Dict[str, Any]:
    """Return a DentNode entry-create draft; never persist or create an entry."""
    _require_internal_key(x_internal_key or x_internal_id)

    def _meter_error() -> None:
        _fire_and_forget(
            report_usage(
                feature="image_to_entry",
                lab_id=body.lab_id,
                user_id=body.user_id,
                model=settings.vision_model,
                status="error",
            )
        )

    try:
        result: ImageToEntryResult = await image_to_entry(
            image_base64=body.image_base64,
            mime_type=body.mime_type or "image/jpeg",
        )
    except CaseExtractionError as exc:
        logger.error(
            "Image-to-entry returned unparseable output",
            extra={"lab_id": body.lab_id, "error": str(exc)},
        )
        _meter_error()
        raise HTTPException(status_code=502, detail=str(exc))
    except OpenRouterError as exc:
        logger.error(
            "Image-to-entry model call failed",
            extra={"lab_id": body.lab_id, "error": str(exc)},
        )
        _meter_error()
        raise HTTPException(status_code=502, detail="Image-to-entry model call failed")

    _fire_and_forget(
        report_usage(
            feature="image_to_entry",
            lab_id=body.lab_id,
            user_id=body.user_id,
            model=result.model,
            usage=result.usage,
            cost=result.cost_usd,
            cost_source="openrouter" if result.cost_usd is not None else "estimated",
            latency_ms=result.latency_ms,
            status="ok",
            meta={
                "work_items": len(result.payload["entry_payload"]["work"]),
                "ready_to_create": result.payload["ready_to_create"],
            },
        )
    )

    return {"success": True, **result.payload, "model": result.model}


@app.post("/rejected-cases-report")
async def rejected_cases_report(
    body: RejectedCasesRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
) -> Dict[str, Any]:
    provided_key = x_internal_key or x_internal_id
    _require_internal_key(provided_key)

    def _meter_error() -> None:
        _fire_and_forget(
            report_usage(
                feature="cron_rejected_cases",
                lab_id=body.lab_id,
                model=settings.vision_model,
                status="error",
            )
        )

    try:
        result = await generate_rejected_cases_report(
            cases=body.cases,
            lab_name=body.lab_name,
            images=[i.model_dump() for i in body.images],
        )
    except RejectedCasesParseError as exc:
        logger.error(
            "Rejected-cases report was not parseable",
            extra={"lab_id": body.lab_id, "error": str(exc)},
        )
        _meter_error()
        raise HTTPException(status_code=502, detail=str(exc))
    except OpenRouterError as exc:
        logger.error(
            "Rejected-cases report model call failed",
            extra={"lab_id": body.lab_id, "error": str(exc)},
        )
        _meter_error()
        raise HTTPException(
            status_code=502, detail="Rejected-cases report model call failed"
        )

    _fire_and_forget(
        report_usage(
            feature="cron_rejected_cases",
            lab_id=body.lab_id,
            model=result.model,
            usage=result.usage,
            cost=result.cost_usd,
            cost_source="openrouter" if result.cost_usd is not None else "estimated",
            latency_ms=result.latency_ms,
            status="ok",
            meta={"cases": len(body.cases), "images": len(body.images)},
        )
    )

    return {"success": True, "report": result.report, "model": result.model}


@app.post("/product-update-email")
async def product_update_email(
    body: ProductUpdateRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
) -> Dict[str, Any]:
    provided_key = x_internal_key or x_internal_id
    _require_internal_key(provided_key)

    try:
        result = await generate_product_update_email(commits=body.commits)
    except OpenRouterError as exc:
        logger.error("Product-update email generation failed", extra={"error": str(exc)})
        _fire_and_forget(
            report_usage(
                feature="cron_product_updates",
                lab_id=PLATFORM_LAB_ID,
                model=settings.model,
                status="error",
            )
        )
        raise HTTPException(
            status_code=502, detail="Product update model call failed"
        )

    _fire_and_forget(
        report_usage(
            feature="cron_product_updates",
            lab_id=PLATFORM_LAB_ID,
            model=result.model,
            usage=result.usage,
            cost=result.cost_usd,
            cost_source="openrouter" if result.cost_usd is not None else "estimated",
            latency_ms=result.latency_ms,
            status="ok",
            meta={"commits": len(body.commits)},
        )
    )

    return {"success": True, "html": result.text, "model": result.model}


@app.exception_handler(HTTPException)
async def http_exc_handler(_request: Request, exc: HTTPException):
    if exc.status_code >= 500:
        logger.error(
            "HTTP exception", extra={"status": exc.status_code, "detail": exc.detail}
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
    )


if __name__ == "__main__":
    import uvicorn

    settings.validate()
    uvicorn.run("server:app", host="0.0.0.0", port=settings.port, reload=False)
