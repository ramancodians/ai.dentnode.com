"""What one voice call cost DentNode, per component.

LiveKit/speech figures are gross list estimates with explicit provenance, not
invoices. OpenRouter generation receipts are preferred when complete. Static
Build/Ship and ElevenLabs API rates were verified 2026-10-25; actual account
plans, allowances, tax and settlement FX remain separate.
"""

import asyncio
import logging
import math
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

import httpx

from agent.config import settings

logger = logging.getLogger(__name__)

PRICE_TABLE_VERSION = "voice-supplier-list-2026-10-25"
PRICE_SOURCES = {
    "livekit": "https://livekit.com/pricing/inference",
    "elevenlabs": "https://elevenlabs.io/pricing/api",
    "openrouter": "https://openrouter.ai/api/v1/models",
    "transport": "https://livekit.com/pricing",
}

# USD per minute of audio.
STT_USD_PER_MIN: Dict[str, float] = {
    "deepgram/nova-3": 0.0048,
    "deepgram/nova-3:multi": 0.0058,
    "deepgram/flux-general-multi": 0.0078,
    "assemblyai/universal-streaming-multilingual": 0.0025,
    "google/gemini-3.5-transcribe-live": 0.0095,
}
# Inference TTS is billed by characters, not generated audio duration.
TTS_USD_PER_1K_CHARS: Dict[str, float] = {
    "fishaudio/s2.1-pro": 0.015,
    "cartesia/sonic-3": 0.05,
    "inworld/inworld-tts-2": 0.025,
    "deepgram/aura-2": 0.03,
}
ELEVENLABS_FLASH_USD_PER_1K_CHARS = 0.04
# LiveKit's SIP leg only. The carrier leg (Plivo) is billed by Plivo and is
# not included here.
SIP_USD_PER_MIN = 0.0040

_OPENROUTER_TTL_SECS = 6 * 3600
_openrouter_prices: Tuple[float, Dict[str, Tuple[float, float]]] = (0.0, {})
_openrouter_retrieved_at: Optional[str] = None
_lock = asyncio.Lock()


def per_minute(table: Dict[str, float], model: str, seconds: float) -> Optional[float]:
    rate = table.get(model)
    return None if rate is None else rate * max(0.0, seconds) / 60.0


async def openrouter_token_prices() -> Dict[str, Tuple[float, float]]:
    """{model: (USD per prompt token, USD per completion token)}, cached 6h.

    List prices from OpenRouter's public model catalog. A fetch failure keeps
    the last good table (or none): cost then reads as unknown, never as zero.
    """
    global _openrouter_prices, _openrouter_retrieved_at
    async with _lock:
        fetched_at, prices = _openrouter_prices
        if prices and time.monotonic() - fetched_at < _OPENROUTER_TTL_SECS:
            return prices
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{settings.openrouter_api_base}/models")
                r.raise_for_status()
            fresh: Dict[str, Tuple[float, float]] = {}
            for m in r.json().get("data", []):
                p = m.get("pricing") or {}
                try:
                    # An absent price is unknown. Explicit numeric/string zero
                    # is retained for actual zero-priced catalog entries.
                    prompt, completion = float(p["prompt"]), float(p["completion"])
                    if math.isfinite(prompt) and math.isfinite(completion) and prompt >= 0 and completion >= 0:
                        fresh[m["id"]] = (prompt, completion)
                except (TypeError, ValueError, KeyError):
                    continue
            _openrouter_prices = (time.monotonic(), fresh)
            _openrouter_retrieved_at = datetime.now(timezone.utc).isoformat()
        except Exception:  # noqa: BLE001 - pricing must never break a call
            logger.warning("OpenRouter price list unavailable; LLM cost left unknown", exc_info=True)
        return _openrouter_prices[1]


def openrouter_price_snapshot() -> dict:
    return {"source": PRICE_SOURCES["openrouter"], "retrieved_at": _openrouter_retrieved_at,
            "cache_ttl_seconds": _OPENROUTER_TTL_SECS,
            "stale": not _openrouter_prices[0] or time.monotonic() - _openrouter_prices[0] >= _OPENROUTER_TTL_SECS}


def llm_cost(prices: Dict[str, Tuple[float, float]], model: str,
             input_tokens: int, output_tokens: int) -> Optional[float]:
    rate = prices.get(model)
    if rate is None:
        return None
    return rate[0] * input_tokens + rate[1] * output_tokens
