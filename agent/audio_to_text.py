"""Audio-to-Text agent: transcribe a Digital Ocean audio URL and summarise it.

A single-call feature agent (no ADK tool loop, no tools) shared by both
app.dentnode.com and d10.live. It sends the downloaded audio to a multimodal
audio model over OpenRouter and returns the verbatim transcript plus an
optional summary. It never queries the database and never dereferences a
non-Digital-Ocean URL — the fetch happens in ``audio_fetch`` with SSRF guards.

The `summary` parameter is a boolean: True (default) asks for a concise summary
in addition to the transcript; False returns the transcript only.
"""

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from .audio_fetch import MODEL_SUPPORTED_FORMATS, AudioFetchError
from .config import settings
from .openrouter import OpenRouterError, _extract_usage, _strip_route_prefix, chat_completion

logger = logging.getLogger(__name__)

# The marker is unlikely to appear in natural speech; splitting on it keeps the
# transcript and summary separate without asking the audio model for JSON.
_SUMMARY_MARKER = "---SUMMARY---"


class UnsupportedAudioFormat(AudioFetchError):
    """The audio is a codec the model cannot transcribe directly."""


@dataclass
class AudioToTextResult:
    transcript: str
    summary: Optional[str]
    model: str
    usage: Dict[str, Any]
    cost_usd: Optional[float]
    latency_ms: int
    audio_format: str
    audio_bytes: int
    segments: List[Dict[str, Any]]
    summary_model: Optional[str] = None
    summary_usage: Optional[Dict[str, Any]] = None
    summary_cost_usd: Optional[float] = None
    summary_latency_ms: Optional[int] = None


class AudioSummaryError(OpenRouterError):
    """Summary failed after a billable transcription completed successfully."""

    def __init__(self, transcription_result: AudioToTextResult):
        super().__init__("Audio summary generation failed")
        self.transcription_result = transcription_result


def _prompt(summary: bool) -> str:
    if summary:
        return (
            "Transcribe the audio verbatim. Then, on a new line, write the marker "
            f'"{_SUMMARY_MARKER}" followed by a concise 2-3 sentence summary of the '
            "key points."
        )
    return "Transcribe the audio verbatim."


def _split(text: str):
    if _SUMMARY_MARKER in text:
        transcript, summary = text.split(_SUMMARY_MARKER, 1)
        return transcript.strip(), summary.strip() or None
    return text.strip(), None


def _validate_format(audio_format: str) -> str:
    """Return the model-compatible format tag, or raise for webm/ogg etc."""
    if audio_format in MODEL_SUPPORTED_FORMATS:
        return audio_format
    raise UnsupportedAudioFormat(
        f"Audio format {audio_format!r} is not supported by the transcription "
        f"model. Transcode to one of {MODEL_SUPPORTED_FORMATS} before uploading."
    )


async def transcribe_audio(
    *,
    audio_bytes: bytes,
    audio_format: str,
    summary: bool,
) -> AudioToTextResult:
    fmt = _validate_format(audio_format)
    b64 = base64.b64encode(audio_bytes).decode()

    if not settings.openrouter_api_key:
        raise OpenRouterError(
            "OpenRouter transcription is not configured",
            code="not_configured",
        )

    # Use OpenRouter's purpose-built STT endpoint rather than Chat Completions.
    # `microsoft/mai-transcribe-2` exposes Azure's diarization capability through
    # OpenRouter, returning recording-local speaker ids on timestamped segments.
    # This remains OpenRouter-only: no provider key or direct provider request is
    # introduced here.
    body: Dict[str, Any] = {
        "model": _strip_route_prefix(settings.audio_to_text_model),
        "input_audio": {"data": b64, "format": fmt},
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment", "word"],
        "provider": {"options": {"azure": {"diarization": {"enabled": True}}}},
    }
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.openrouter_site_url,
        "X-Title": settings.openrouter_app_name,
    }
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=settings.audio_to_text_timeout_secs) as client:
            response = await client.post(
                f"{settings.openrouter_api_base}/audio/transcriptions",
                json=body,
                headers=headers,
            )
    except httpx.HTTPError as exc:
        raise OpenRouterError(
            "OpenRouter transcription failed",
            code="transport_error",
        ) from exc
    latency_ms = int((time.monotonic() - t0) * 1000)
    if response.status_code >= 400:
        raise OpenRouterError(
            "OpenRouter transcription failed",
            code="provider_http_error",
            status_code=response.status_code,
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise OpenRouterError(
            "OpenRouter transcription returned an invalid response",
            code="invalid_response",
            status_code=response.status_code,
        ) from exc

    transcript = str(data.get("text") or "").strip()
    if not transcript:
        raise OpenRouterError(
            "OpenRouter transcription returned an empty transcript",
            code="empty_transcript",
        )
    raw_usage = data.get("usage") or {}
    cost_usd: Optional[float] = None
    try:
        if raw_usage.get("cost") is not None:
            cost_usd = float(raw_usage["cost"])
    except (TypeError, ValueError):
        pass
    segments = [segment for segment in (data.get("segments") or []) if isinstance(segment, dict)]

    result = AudioToTextResult(
        transcript=transcript,
        summary=None,
        model=str(data.get("model") or body["model"]),
        usage=_extract_usage(raw_usage),
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        audio_format=fmt,
        audio_bytes=len(audio_bytes),
        segments=segments,
    )

    if summary:
        try:
            summary_result = await chat_completion(
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarise this dental visit transcript in 2-3 concise sentences. "
                        "Do not add facts that are not in the transcript.\n\n" + transcript
                    ),
                }],
                model=settings.model,
                temperature=0.1,
                timeout_secs=settings.audio_to_text_timeout_secs,
            )
        except OpenRouterError as exc:
            raise AudioSummaryError(result) from exc
        result.summary = summary_result.text.strip() or None
        result.summary_model = summary_result.model
        result.summary_usage = summary_result.usage
        result.summary_cost_usd = summary_result.cost_usd
        result.summary_latency_ms = summary_result.latency_ms
    return result
