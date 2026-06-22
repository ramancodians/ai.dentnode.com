"""FastAPI entrypoint for the Laby ADK agent service.

Endpoints:
  GET  /health         — liveness/readiness (no auth).
  POST /agent/run      — run one turn; streams normalized NDJSON events.

This service is internal-only. The Node backend calls /agent/run with the
shared x-internal-key. It is deployed to Cloud Run with --ingress internal.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.config import settings
from agent.logging_setup import setup_logging
from agent.runner import run_turn

# Wire JSON logging before the first log line is emitted.
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.validate()
    logger.info("Laby agent started", extra={"model": settings.model, "port": settings.port})
    yield


app = FastAPI(title="Laby ADK Agent", version="1.0.0", lifespan=lifespan)


class HistoryTurn(BaseModel):
    role: str
    text: str


class RunRequest(BaseModel):
    lab_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    history: Optional[List[HistoryTurn]] = None


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
        "provider": "deepseek",
    }


@app.post("/agent/run")
async def agent_run(
    body: RunRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
) -> StreamingResponse:
    _require_internal_key(x_internal_key)

    history = [t.model_dump() for t in (body.history or [])]
    t0 = time.monotonic()

    async def event_stream():
        try:
            async for event in run_turn(
                lab_id=body.lab_id,
                user_id=body.user_id,
                question=body.question,
                history=history,
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
