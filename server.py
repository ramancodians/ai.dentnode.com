"""FastAPI entrypoint for the Laby ADK agent service.

Endpoints:
  GET  /health         — liveness/readiness (no auth).
  POST /agent/run      — run one turn; streams normalized NDJSON events.

This service is internal-only. The Node backend calls /agent/run with the
shared x-internal-key or x-internal-id. It is deployed to Cloud Run with --ingress all.
"""

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from agent.audio_fetch import AudioFetchError, fetch_audio
from agent.audio_to_text import UnsupportedAudioFormat, transcribe_audio
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
from agent.text_to_speech import (
    SUPPORTED_FORMATS,
    SUPPORTED_VOICES,
    SpeechMeta,
    SpeechStream,
    media_type_for,
    validate_request as validate_speech_request,
)
from agent.browser_runner import run_browser_turn
from agent.scan_review import generate_scan_review
from agent.usage import report_usage
from agent.d10 import D10RequestContext, UsageOutbox, run_d10_turn

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

d10_usage_outbox = UsageOutbox(settings.d10_usage_outbox_path)


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
    await d10_usage_outbox.initialize()
    d10_outbox_task = asyncio.create_task(
        d10_usage_outbox.run(settings.d10_usage_flush_interval_secs)
    )
    try:
        yield
    finally:
        d10_usage_outbox.stop()
        try:
            await asyncio.wait_for(d10_outbox_task, timeout=5.0)
        except asyncio.TimeoutError:
            d10_outbox_task.cancel()
        # Best effort only: records are already committed locally and will be
        # retried on the next startup if D10 is temporarily unavailable.
        await d10_usage_outbox.flush_once()


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


class D10HistoryTurn(BaseModel):
    role: str
    text: str = Field(..., min_length=1)


class D10RunRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=16000)
    context: D10RequestContext
    history: Optional[List[D10HistoryTurn]] = Field(default=None, max_length=40)


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


class AudioToTextRequest(BaseModel):
    """Shared by app.dentnode.com and d10.live. `summary` (default True) adds a
    concise summary to the verbatim transcript; the audio must be a Digital
    Ocean object URL."""

    audio_url: str = Field(..., min_length=1, max_length=2048)
    summary: bool = True
    lab_id: Optional[str] = None
    user_id: Optional[str] = None


class TextToSpeechRequest(BaseModel):
    """Open to every trusted internal caller (app.dentnode.com, d10.live, …).

    The upper bound on `text` is enforced against TEXT_TO_SPEECH_MAX_CHARS in
    the handler; the field cap here is only a cheap guard against a caller
    posting a megabyte of prose.
    """

    text: str = Field(..., min_length=1, max_length=100_000)
    # Falls back to TEXT_TO_SPEECH_VOICE when the caller does not care.
    voice: Optional[str] = None
    audio_format: str = "wav"
    # Streaming is the default: the first bytes leave here well under a second
    # after the request, so a caller can start playback while the rest arrives.
    # Set false to get one buffered file with a Content-Length and an
    # exact-size WAV header, which is what a caller that is about to upload the
    # result somewhere actually wants.
    stream: bool = True
    lab_id: Optional[str] = None
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


def _require_d10_internal_key(provided: Optional[str]) -> None:
    if not settings.d10_internal_key:
        raise HTTPException(status_code=500, detail="D10_INTERNAL_KEY not configured")
    if not provided or provided != settings.d10_internal_key:
        raise HTTPException(status_code=401, detail="Invalid D10 internal key")


def _require_shared_internal_key(
    provided_internal: Optional[str], provided_d10: Optional[str]
) -> None:
    """Accept either the app.dentnode.com key or the D10 key.

    The Audio-to-Text agent is shared between the two services, so either
    trusted caller may invoke it.
    """
    if settings.internal_key and provided_internal == settings.internal_key:
        return
    if settings.d10_internal_key and provided_d10 == settings.d10_internal_key:
        return
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


@app.post("/d10/agent/run")
async def d10_agent_run(
    body: D10RunRequest,
    x_d10_internal_key: Optional[str] = Header(
        default=None, alias="x-d10-internal-key"
    ),
) -> StreamingResponse:
    """Run one D10 Agent turn with D10-asserted identity and correlation."""
    _require_d10_internal_key(x_d10_internal_key)
    history = [turn.model_dump() for turn in (body.history or [])]

    async def event_stream():
        async for event in run_d10_turn(
            message=body.message,
            context=body.context,
            history=history,
            outbox=d10_usage_outbox,
        ):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Correlation-ID": body.context.correlation_id,
        },
    )


@app.post("/audio-to-text")
async def audio_to_text(
    body: AudioToTextRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
    x_d10_internal_key: Optional[str] = Header(default=None, alias="x-d10-internal-key"),
) -> Dict[str, Any]:
    """Transcribe + summarise a Digital Ocean audio URL.

    Shared by app.dentnode.com (x-internal-key) and d10.live
    (x-d10-internal-key). Only Digital Ocean URLs are accepted.
    """
    _require_shared_internal_key(x_internal_key or x_internal_id, x_d10_internal_key)

    try:
        audio = await fetch_audio(body.audio_url)
    except AudioFetchError as exc:
        logger.warning("Audio fetch rejected", extra={"error": str(exc)})
        raise HTTPException(status_code=400, detail=str(exc))

    lab_id = body.lab_id or PLATFORM_LAB_ID

    try:
        result = await transcribe_audio(
            audio_bytes=audio.data,
            audio_format=audio.format,
            summary=body.summary,
        )
    except UnsupportedAudioFormat as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except OpenRouterError as exc:
        logger.error(
            "Audio-to-text generation failed",
            extra={"lab_id": lab_id, "error": str(exc)},
        )
        _fire_and_forget(
            report_usage(
                feature="audio_to_text",
                lab_id=lab_id,
                user_id=body.user_id,
                model=settings.audio_to_text_model,
                status="error",
            )
        )
        raise HTTPException(status_code=502, detail="Audio-to-text model call failed")

    _fire_and_forget(
        report_usage(
            feature="audio_to_text",
            lab_id=lab_id,
            user_id=body.user_id,
            model=result.model,
            usage=result.usage,
            cost=result.cost_usd,
            cost_source="openrouter" if result.cost_usd is not None else "estimated",
            latency_ms=result.latency_ms,
            status="ok",
            meta={
                "audio_bytes": result.audio_bytes,
                "audio_format": result.audio_format,
                "speaker_segments": len(result.segments),
            },
        )
    )

    return {
        "success": True,
        "transcript": result.transcript,
        "summary": result.summary,
        "model": result.model,
        "audio_format": result.audio_format,
        "audio_bytes": result.audio_bytes,
        "segments": result.segments,
    }


# Keeps X-TTS-Transcript-B64 inside a typical 8 KB per-header proxy limit even
# when every character is multi-byte.
_TRANSCRIPT_HEADER_MAX_CHARS = 1024


def _speech_headers(meta: SpeechMeta, *, streaming: bool) -> Dict[str, str]:
    """Metadata sidecar for a binary body.

    The response body is audio, so everything a caller might want to log,
    display or bill against has to ride in headers. They are ASCII-safe by
    construction except the transcript, which is base64 so a non-Latin-1 script
    (Hindi, say) cannot make the response unencodable.
    """
    headers = {
        "X-TTS-Model": meta.model,
        "X-TTS-Voice": meta.voice,
        "X-TTS-Format": meta.audio_format,
        "X-TTS-Sample-Rate": str(meta.sample_rate),
        "X-TTS-Channels": str(meta.channels),
        "Content-Disposition": f'attachment; filename="speech.{meta.audio_format}"',
        "Cache-Control": "no-store",
    }
    if streaming:
        # Nothing below is known until the stream drains, so a streaming
        # response cannot carry it — the caller gets it from the ledger.
        #
        # The flag is worth stating because a chunked WAV is not byte-identical
        # to a buffered one: its two RIFF size fields hold the streaming
        # sentinel, since the length is unknown when the header goes out.
        # ffprobe, browsers and ffmpeg all read such a file correctly, but
        # anything that trusts the size field verbatim (Python's `wave`, some
        # metadata scrapers) will report a nonsense duration. A caller that
        # needs exact metadata on the file itself should send stream=false.
        headers["X-TTS-Streaming"] = "chunked"
        headers["X-Accel-Buffering"] = "no"
        return headers
    headers.update({
        "X-TTS-Audio-Ms": str(meta.duration_ms),
        "X-TTS-Latency-Ms": str(meta.latency_ms),
        "X-TTS-Cost-Usd": f"{meta.cost_usd:.8f}" if meta.cost_usd is not None else "",
        # What the model actually said. Compare it against the text you sent to
        # detect the one real failure mode of using a chat model as a TTS
        # engine: it answering the text instead of reading it.
    })
    # Header budget: 4000 characters of Devanagari is ~12 KB of UTF-8 and ~16 KB
    # of base64, past what most proxies accept in a single header. Truncating
    # the source keeps the response deliverable; a deviation shows up at the
    # start of the transcript anyway, which is what this header is for.
    transcript = meta.transcript
    truncated = len(transcript) > _TRANSCRIPT_HEADER_MAX_CHARS
    if truncated:
        transcript = transcript[:_TRANSCRIPT_HEADER_MAX_CHARS]
        headers["X-TTS-Transcript-Truncated"] = "true"
    headers["X-TTS-Transcript-B64"] = base64.b64encode(
        transcript.encode("utf-8")
    ).decode("ascii")
    return headers


@app.post("/text-to-speech")
async def text_to_speech(
    body: TextToSpeechRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
    x_d10_internal_key: Optional[str] = Header(default=None, alias="x-d10-internal-key"),
):
    """Synthesise speech from text and return the audio bytes.

    Open to any trusted internal caller — app.dentnode.com (x-internal-key) or
    d10.live (x-d10-internal-key) — like /audio-to-text.

    Nothing is stored here. The audio is streamed straight back to the caller,
    which owns the decision of whether it lands in Spaces, goes out over
    WhatsApp, or is played once and dropped.
    """
    _require_shared_internal_key(x_internal_key or x_internal_id, x_d10_internal_key)

    voice = body.voice or settings.text_to_speech_voice
    try:
        text = validate_speech_request(
            text=body.text, voice=voice, audio_format=body.audio_format
        )
    except ValueError as exc:
        # Covers UnsupportedVoice/UnsupportedFormat too — both subclass ValueError.
        raise HTTPException(status_code=400, detail=str(exc))

    lab_id = body.lab_id or PLATFORM_LAB_ID

    def meter(meta: SpeechMeta) -> None:
        _fire_and_forget(
            report_usage(
                feature="text_to_speech",
                lab_id=lab_id,
                user_id=body.user_id,
                model=meta.model,
                usage=meta.usage,
                cost=meta.cost_usd,
                cost_source=meta.cost_source,
                latency_ms=meta.latency_ms,
                status=meta.status,
                meta=meta.as_usage_meta(),
            )
        )

    # open() blocks until the first audio frame arrives, so an upstream failure
    # becomes a clean 502 here rather than a truncated file the caller has
    # already started saving.
    try:
        speech = await SpeechStream(
            text=text, voice=voice, audio_format=body.audio_format
        ).open()
    except OpenRouterError as exc:
        logger.error(
            "Text-to-speech generation failed",
            extra={"lab_id": lab_id, "error": str(exc)},
        )
        _fire_and_forget(
            report_usage(
                feature="text_to_speech",
                lab_id=lab_id,
                user_id=body.user_id,
                model=settings.text_to_speech_model,
                status="error",
                meta={"voice": voice, "chars": len(text)},
            )
        )
        raise HTTPException(status_code=502, detail="Text-to-speech model call failed")

    if not body.stream:
        audio = await speech.collect()
        meter(speech.meta)
        if speech.meta.status != "ok":
            # Nothing has been sent yet on this path, so a mid-stream failure
            # can still surface as an error instead of a silently truncated
            # file. Returning the partial audio as a 200 would defeat the only
            # reason a caller asks for the buffered mode.
            logger.error(
                "Text-to-speech stream ended early",
                extra={"lab_id": lab_id, "audio_bytes": speech.meta.audio_bytes},
            )
            raise HTTPException(
                status_code=502, detail="Text-to-speech stream ended early"
            )
        return Response(
            content=audio,
            media_type=media_type_for(speech.meta.audio_format),
            headers=_speech_headers(speech.meta, streaming=False),
        )

    async def audio_stream():
        try:
            async for chunk in speech.chunks():
                yield chunk
        finally:
            # Runs on a client disconnect as well as on a clean finish, so the
            # spend OpenRouter has already charged for is never lost.
            meter(speech.meta)

    return StreamingResponse(
        audio_stream(),
        media_type=media_type_for(speech.meta.audio_format),
        headers=_speech_headers(speech.meta, streaming=True),
    )


@app.get("/text-to-speech/voices")
async def text_to_speech_voices(
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
    x_internal_id: Optional[str] = Header(default=None, alias="x-internal-id"),
    x_d10_internal_key: Optional[str] = Header(default=None, alias="x-d10-internal-key"),
) -> Dict[str, Any]:
    """What a caller may put in `voice` / `audio_format`, and the current default."""
    _require_shared_internal_key(x_internal_key or x_internal_id, x_d10_internal_key)
    return {
        "voices": list(SUPPORTED_VOICES),
        "formats": list(SUPPORTED_FORMATS),
        "default_voice": settings.text_to_speech_voice,
        "model": settings.text_to_speech_model,
        "max_chars": settings.text_to_speech_max_chars,
    }


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
