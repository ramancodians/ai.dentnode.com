"""Routes for the Voice Lab dev harness."""

import asyncio
import datetime
import hmac
import json
import logging
import pathlib
import secrets
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.config import settings
from agent.usage import report_usage
from agent.voice_agent import (
    DEFAULT_OBJECTIVE,
    LANGUAGES,
    REPLY_MODELS,
    STT_MODELS,
    TTS_MODELS,
    TTS_VOICES,
    VoiceTurnRequest,
    run_voice_turn,
)
from voice_call import config as lk_config

logger = logging.getLogger(__name__)

router = APIRouter()

_PAGE = pathlib.Path(__file__).with_name("index.html")
_MAX_AUDIO_BYTES = 10 * 1024 * 1024
_PLATFORM_LAB_ID = "__platform__"
_background: set = set()


def _require_key(provided: Optional[str]) -> None:
    expected = settings.internal_key
    if not expected or not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid internal key")


def _meter(model: str, cost: Any, latency_ms: Any, usage: Optional[Dict[str, Any]] = None) -> None:
    if cost is None and latency_ms is None:
        return
    task = asyncio.create_task(report_usage(
        feature="voice_lab", lab_id=_PLATFORM_LAB_ID, model=model, usage=usage,
        cost=cost, cost_source="openrouter" if cost is not None else "estimated",
        latency_ms=latency_ms,
    ))
    _background.add(task)
    task.add_done_callback(_background.discard)


@router.get("/voice-lab", include_in_schema=False)
async def voice_lab_page() -> FileResponse:
    # The page is static and carries no secret; every call it makes needs the key.
    return FileResponse(_PAGE, media_type="text/html", headers={"Cache-Control": "no-store"})


@router.get("/voice-lab/options")
async def voice_lab_options(
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
) -> Dict[str, Any]:
    _require_key(x_internal_key)
    return {
        "stt_models": list(STT_MODELS),
        "reply_models": list(REPLY_MODELS),
        "tts_models": list(TTS_MODELS),
        "tts_voices": {k: list(v) for k, v in TTS_VOICES.items()},
        "languages": list(LANGUAGES),
        "default_objective": DEFAULT_OBJECTIVE,
        "livekit": {
            "enabled": bool(settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret),
            "stt_models": list(lk_config.STT_MODELS),
            "reply_models": list(REPLY_MODELS),
            "tts_models": list(lk_config.TTS_MODELS),
        },
    }


class LiveKitSessionRequest(BaseModel):
    stt_model: str = lk_config.STT_MODELS[0]
    reply_model: str = "anthropic/claude-haiku-4.5"
    tts_model: str = lk_config.TTS_MODELS[0]
    language: str = ""
    objective: str = Field(default="", max_length=1000)


@router.post("/voice-lab/livekit/session")
async def voice_lab_livekit_session(
    body: LiveKitSessionRequest,
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
) -> Dict[str, str]:
    """A fresh LiveKit room with the voice agent dispatched into it by name.

    The browser joins with the returned token; the dispatch rides on the token,
    so the agent is only started when that participant actually connects.
    """
    _require_key(x_internal_key)
    if not (settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret):
        raise HTTPException(status_code=503, detail="LiveKit is not configured")
    if (body.stt_model not in lk_config.STT_MODELS or body.reply_model not in REPLY_MODELS
            or body.tts_model not in lk_config.TTS_MODELS or body.language not in lk_config.LANGUAGES):
        raise HTTPException(status_code=400, detail="Unknown model or language")

    from livekit import api  # optional dependency path: only needed for this route

    room = f"voicelab-{secrets.token_hex(6)}"
    metadata = json.dumps({
        "source": "voice_lab", "objective": body.objective.strip() or lk_config.DEFAULT_OBJECTIVE,
        "language": body.language, "stt_model": body.stt_model,
        "reply_model": body.reply_model, "tts_model": body.tts_model,
    })
    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(f"tester-{secrets.token_hex(4)}")
        .with_name("Voice Lab tester")
        .with_ttl(datetime.timedelta(minutes=15))
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .with_room_config(api.RoomConfiguration(agents=[
            api.RoomAgentDispatch(agent_name=lk_config.AGENT_NAME, metadata=metadata)
        ]))
        .to_jwt()
    )
    return {"url": settings.livekit_url, "token": token, "room": room}


@router.post("/voice-lab/turn")
async def voice_lab_turn(
    audio: Optional[UploadFile] = File(default=None),
    text: str = Form(default="", max_length=1000),
    history: str = Form(default="[]", max_length=20000),
    objective: str = Form(default="", max_length=1000),
    language: str = Form(default=""),
    stt_model: str = Form(default=STT_MODELS[0]),
    reply_model: str = Form(default=REPLY_MODELS[0]),
    tts_model: str = Form(default=TTS_MODELS[0]),
    voice: str = Form(default=""),
    style: str = Form(default="", max_length=200),
    x_internal_key: Optional[str] = Header(default=None, alias="x-internal-key"),
) -> StreamingResponse:
    """One conversational turn, streamed back as NDJSON events."""
    _require_key(x_internal_key)
    if stt_model not in STT_MODELS or reply_model not in REPLY_MODELS or tts_model not in TTS_MODELS:
        raise HTTPException(status_code=400, detail="Unknown model")
    if voice not in TTS_VOICES[tts_model]:
        raise HTTPException(status_code=400, detail="Unknown voice for this model")
    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail="Unknown language")
    try:
        turns: List[Dict[str, str]] = [
            {"caller": str(t.get("caller", ""))[:1000], "agent": str(t.get("agent", ""))[:900]}
            for t in json.loads(history) if isinstance(t, dict)
        ]
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="history must be a JSON list")

    audio_bytes: Optional[bytes] = None
    if audio is not None:
        audio_bytes = await audio.read(_MAX_AUDIO_BYTES + 1)
        if len(audio_bytes) > _MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio is too large")
        # The page records WAV itself so no transcoding is needed server-side.
        if not (audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"):
            raise HTTPException(status_code=400, detail="Audio must be WAV")
    if not audio_bytes and not text.strip():
        raise HTTPException(status_code=400, detail="Send audio or text")

    req = VoiceTurnRequest(
        audio=audio_bytes, audio_format="wav", text=text, history=turns,
        objective=objective.strip(), language=language, stt_model=stt_model,
        reply_model=reply_model, tts_model=tts_model, voice=voice, style=style.strip(),
    )

    async def events():
        timings: Dict[str, Any] = {}
        async for event in run_voice_turn(req):
            if event["type"] == "done":
                timings = event["timings"]
            elif event["type"] == "cost":
                costs = event["costs"]
                if audio_bytes:
                    _meter(stt_model, costs.get("stt_usd"), timings.get("stt_ms"))
                _meter(reply_model, costs.get("reply_usd"), timings.get("reply_ms"),
                       costs.get("reply_usage"))
                _meter(tts_model, costs.get("tts_usd"), timings.get("tts_total_ms"))
            elif event["type"] == "error":
                logger.warning("Voice lab turn failed", extra={"stage": event["stage"]})
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
