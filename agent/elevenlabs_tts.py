"""ElevenLabs text-to-speech for phone calls: the one voice every DentNode call speaks with.

Plivo plays the returned MP3 with <Play>, so its own <Speak>/Polly voice is not
used. Same key and Hindi voice as the LiveKit worker (ELEVEN_API_KEY,
ELEVENLABS_HINDI_VOICE_ID); Flash v2.5 reads Hinglish (Devanagari Hindi mixed
with English words) naturally and is the low-latency model.
"""

import os
import time
from typing import Optional, Tuple

import httpx

MODEL_ID = "eleven_flash_v2_5"
# 22.05 kHz mono MP3 is well inside what Plivo <Play> accepts and keeps files small.
OUTPUT_FORMAT = "mp3_22050_32"
# Mirrors voice_call/costs.py; list price, not provider-reconciled usage.
USD_PER_1K_CHARS = 0.04


class TtsUnavailable(Exception):
    pass


def tts_ready() -> bool:
    return bool(os.environ.get("ELEVEN_API_KEY") and os.environ.get("ELEVENLABS_HINDI_VOICE_ID"))


async def synthesize(text: str, voice_id: Optional[str] = None) -> Tuple[bytes, int]:
    """Return (mp3 bytes, latency ms). Raises TtsUnavailable on any failure."""
    key = os.environ.get("ELEVEN_API_KEY")
    voice = voice_id or os.environ.get("ELEVENLABS_HINDI_VOICE_ID")
    if not key or not voice:
        raise TtsUnavailable("ElevenLabs is not configured")
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
                params={"output_format": OUTPUT_FORMAT},
                headers={"xi-api-key": key, "Accept": "audio/mpeg"},
                json={"text": text, "model_id": MODEL_ID,
                      "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
            )
    except httpx.HTTPError as exc:
        raise TtsUnavailable(type(exc).__name__) from exc
    if response.status_code != 200 or not response.content:
        raise TtsUnavailable(f"ElevenLabs returned {response.status_code}")
    return response.content, int((time.monotonic() - started) * 1000)
