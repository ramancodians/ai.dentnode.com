"""Voice agent: one spoken turn — caller audio → transcript → reply → speech.

Built to measure what a phone conversation on D10 would feel like before any
calling code depends on it. Each stage is its own OpenRouter call, timed on its
own, so the latency budget is visible stage by stage:

  1. STT    POST /audio/transcriptions   (caller audio → text)
  2. Reply  POST /chat/completions       (text + history → short reply)
  3. TTS    POST /audio/speech           (reply → MP3, streamed)

TTS goes through OpenRouter's dedicated speech endpoint, not the chat-model
path in ``text_to_speech.py``: a real TTS model reads its input verbatim, where
``gpt-audio-mini`` translated and truncated patient scripts. Output is raw PCM
because Gemini TTS rejects ``mp3``; the rate differs per model (Gemini and MAI
24 kHz, Fish 44.1 kHz), so it is read off the response Content-Type and sent
ahead of the audio.

``run_voice_turn`` yields events as each stage lands so a caller can show the
transcript before the reply exists and start playing audio on its first chunk
— which is the number that matters on a call (time to first audio).

Defaults live here rather than in env vars: this is a dev harness, and every
option is also passed per request.
"""

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .calling_prompt import calling_system_prompt
from .config import settings
from .openrouter import OpenRouterError, _strip_route_prefix, chat_completion

logger = logging.getLogger(__name__)

STT_MODELS = (
    "openai/gpt-4o-mini-transcribe",
    "openai/whisper-large-v3-turbo",
    "openai/gpt-4o-transcribe",
    "microsoft/mai-transcribe-2",
)
REPLY_MODELS = (
    _strip_route_prefix(settings.d10_agent_model),
    "google/gemini-3.5-flash-lite",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-haiku-4.5",
)
# These reject `reasoning: {enabled: false}` with a 400 ("Reasoning is mandatory").
_REASONING_MANDATORY = frozenset({"google/gemini-3.5-flash-lite"})
TTS_MODELS = (
    "google/gemini-3.8-flash-tts",
    "google/gemini-3.8-flash-lite-tts",
    "microsoft/mai-voice-2",
    "microsoft/mai-voice-2-flash",
    "fish-audio/s2.1-pro",
)
# Voice ids are provider-specific; the first entry is the default per model.
TTS_VOICES: Dict[str, tuple] = {
    "google/gemini-3.8-flash-tts": ("Kore", "Aoede", "Leda", "Achird", "Charon", "Puck"),
    "google/gemini-3.8-flash-lite-tts": ("Kore", "Aoede", "Leda", "Achird", "Charon", "Puck"),
    "microsoft/mai-voice-2": ("en-US-Harper:MAI-Voice-2",),
    "microsoft/mai-voice-2-flash": ("en-US-Harper:MAI-Voice-2",),
    "fish-audio/s2.1-pro": ("",),
}
LANGUAGES = ("", "en", "hi", "mr")

DEFAULT_OBJECTIVE = (
    "Remind the patient of their appointment with Dr. Mehta tomorrow at 11:30 AM "
    "and ask them to confirm."
)

_TIMEOUT_SECS = 30.0

# One pooled client: a fresh AsyncClient per turn cost ~250 ms of TLS setup
# before any model was called, which would be measured as model latency.
_client: Optional[httpx.AsyncClient] = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=_TIMEOUT_SECS)
    return _client


@dataclass
class VoiceTurnRequest:
    audio: Optional[bytes]
    audio_format: str
    text: str
    history: List[Dict[str, str]]
    objective: str
    language: str
    stt_model: str
    reply_model: str
    tts_model: str
    voice: str
    style: str


def _headers() -> Dict[str, str]:
    if not settings.openrouter_api_key:
        raise OpenRouterError("OpenRouter is not configured", code="not_configured")
    return {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.openrouter_site_url,
        "X-Title": settings.openrouter_app_name,
    }


_STATUS_HINTS = {
    401: "OpenRouter key rejected",
    402: "OpenRouter credits exhausted",
    404: "model or voice not found",
    429: "rate limited",
}


def _describe(stage: str, status_code: Optional[int]) -> str:
    # Status only: OpenRouterError messages must never carry provider bodies.
    if status_code is None:
        return f"{stage} failed"
    hint = _STATUS_HINTS.get(status_code, "upstream error")
    return f"{stage} failed: {hint} (HTTP {status_code})"


def _upstream_error(stage: str, response: httpx.Response) -> OpenRouterError:
    return OpenRouterError(
        _describe(stage, response.status_code),
        code="provider_http_error",
        status_code=response.status_code,
    )


def _pcm_format(content_type: str) -> Dict[str, int]:
    """Parse ``audio/pcm;rate=24000;channels=1``; 16-bit little-endian is implied."""
    fmt = {"rate": 24000, "channels": 1}
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key in fmt and value.isdigit():
            fmt[key] = int(value)
    return fmt


async def _transcribe(client: httpx.AsyncClient, req: VoiceTurnRequest) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": req.stt_model,
        "input_audio": {"data": base64.b64encode(req.audio or b"").decode(), "format": req.audio_format},
    }
    if req.language:
        body["language"] = req.language
    t0 = time.monotonic()
    response = await client.post(
        f"{settings.openrouter_api_base}/audio/transcriptions", json=body, headers=_headers()
    )
    ms = int((time.monotonic() - t0) * 1000)
    if response.status_code >= 400:
        raise _upstream_error("Transcription", response)
    data = response.json()
    cost = (data.get("usage") or {}).get("cost")
    return {"text": str(data.get("text") or "").strip(), "ms": ms, "cost_usd": cost}


async def _generation_cost(client: httpx.AsyncClient, generation_id: str) -> Optional[float]:
    """TTS cost is only on /generation, which lags the response by a second or two."""
    for _ in range(5):
        await asyncio.sleep(1.5)
        try:
            r = await client.get(
                f"{settings.openrouter_api_base}/generation",
                params={"id": generation_id},
                headers=_headers(),
            )
        except httpx.HTTPError:
            continue
        if r.status_code == 200:
            cost = (r.json().get("data") or {}).get("total_cost")
            return float(cost) if cost is not None else None
    return None


async def run_voice_turn(req: VoiceTurnRequest) -> AsyncIterator[Dict[str, Any]]:
    """Yield ``transcript`` → ``reply`` → ``format`` → ``audio``* → ``done`` → ``cost``.

    Any failure yields a single ``error`` event carrying the stage and stops.
    Timings are server-side wall clock in milliseconds from turn start.
    """
    started = time.monotonic()

    def since_start() -> int:
        return int((time.monotonic() - started) * 1000)

    timings: Dict[str, Any] = {}
    costs: Dict[str, Any] = {}

    client = _http()
    # 1. STT — skipped when the turn was typed.
    heard = req.text.strip()
    if req.audio:
        try:
            stt = await _transcribe(client, req)
        except OpenRouterError as exc:
            yield {"type": "error", "stage": "stt", "message": str(exc)}
            return
        except httpx.HTTPError:
            yield {"type": "error", "stage": "stt", "message": "Transcription failed: network error"}
            return
        heard = stt["text"]
        timings["stt_ms"] = stt["ms"]
        costs["stt_usd"] = stt["cost_usd"]
    if not heard:
        yield {"type": "error", "stage": "stt", "message": "Nothing was heard"}
        return
    yield {"type": "transcript", "text": heard, "at_ms": since_start()}

    # 2. Reply.
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": calling_system_prompt(req.objective or DEFAULT_OBJECTIVE, multilingual=True)}
    ]
    for turn in req.history[-8:]:
        messages.append({"role": "user", "content": turn.get("caller", "")})
        messages.append({"role": "assistant", "content": turn.get("agent", "")})
    messages.append({"role": "user", "content": heard})
    try:
        result = await chat_completion(
            messages=messages, model=req.reply_model, temperature=0.3,
            max_tokens=160, timeout_secs=_TIMEOUT_SECS,
            # Hidden reasoning costs ~0.5s before the first word and can eat
            # the whole token cap, leaving an empty reply.
            extra_body=(None if req.reply_model in _REASONING_MANDATORY
                        else {"reasoning": {"enabled": False}}),
        )
    except OpenRouterError as exc:
        yield {"type": "error", "stage": "reply", "message": _describe("Reply", exc.status_code)}
        return
    reply = result.text.strip()
    if not reply:
        yield {"type": "error", "stage": "reply", "message": "Model returned no reply"}
        return
    timings["reply_ms"] = result.latency_ms
    costs["reply_usd"] = result.cost_usd
    yield {"type": "reply", "text": reply, "model": result.model, "at_ms": since_start()}

    # 3. TTS, streamed through as it arrives.
    body: Dict[str, Any] = {"model": req.tts_model, "input": reply, "response_format": "pcm"}
    if req.voice:
        body["voice"] = req.voice
    if req.style and req.tts_model.startswith("google/"):
        body["provider"] = {"options": {"google-ai-studio": {"speech_metadata": {"style": req.style}}}}
    tts_started = time.monotonic()
    generation_id = None
    audio_bytes = 0
    try:
        async with client.stream(
            "POST", f"{settings.openrouter_api_base}/audio/speech", json=body, headers=_headers()
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise _upstream_error("Speech", response)
            generation_id = response.headers.get("x-generation-id")
            yield {"type": "format", **_pcm_format(response.headers.get("content-type", ""))}
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                if "tts_first_byte_ms" not in timings:
                    timings["tts_first_byte_ms"] = int((time.monotonic() - tts_started) * 1000)
                    timings["first_audio_ms"] = since_start()
                audio_bytes += len(chunk)
                yield {"type": "audio", "b64": base64.b64encode(chunk).decode("ascii")}
    except OpenRouterError as exc:
        yield {"type": "error", "stage": "tts", "message": str(exc)}
        return
    except httpx.HTTPError:
        yield {"type": "error", "stage": "tts", "message": "Speech failed: network error"}
        return
    timings["tts_total_ms"] = int((time.monotonic() - tts_started) * 1000)
    timings["total_ms"] = since_start()
    yield {"type": "done", "timings": timings, "audio_bytes": audio_bytes,
           "reply_chars": len(reply)}

    if generation_id:
        costs["tts_usd"] = await _generation_cost(client, generation_id)
    costs["reply_usage"] = result.usage
    yield {"type": "cost", "costs": costs}
