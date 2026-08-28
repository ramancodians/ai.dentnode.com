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
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .audio_fetch import MODEL_SUPPORTED_FORMATS, AudioFetchError
from .config import settings
from .openrouter import OpenRouterError, chat_completion

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

    result = await chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _prompt(summary)},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64, "format": fmt},
                    },
                ],
            }
        ],
        model=settings.audio_to_text_model,
        temperature=0.1,
        timeout_secs=settings.audio_to_text_timeout_secs,
    )

    transcript, summary_text = _split(result.text)
    return AudioToTextResult(
        transcript=transcript,
        summary=summary_text if summary else None,
        model=result.model,
        usage=result.usage,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        audio_format=fmt,
        audio_bytes=len(audio_bytes),
    )
