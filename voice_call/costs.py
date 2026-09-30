"""What one voice call cost DentNode, per component.

LiveKit Inference reports usage (audio seconds, characters) but not money, and
the OpenAI-compatible plugin does not surface OpenRouter's per-request cost. So
cost here is list price x usage, and every figure is tagged
``cost_source="price_table"`` with ``PRICE_TABLE_VERSION`` so the owner report
can tell it apart from a provider-reported cost.

LiveKit prices: livekit.com/pricing, Build/Ship plan, read 2026-10-01. Update
the version string whenever a number changes.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple

import httpx

from agent.config import settings

logger = logging.getLogger(__name__)

PRICE_TABLE_VERSION = "livekit-build-2026-10-01"

# USD per minute of audio.
STT_USD_PER_MIN: Dict[str, float] = {
    # Nova-3 list price is for monolingual; we run it in "multi", which Deepgram
    # prices higher, so this under-states STT slightly.
    "deepgram/nova-3": 0.0048,
    "deepgram/flux-general-multi": 0.0065,
    "assemblyai/universal-streaming-multilingual": 0.0025,
    "google/gemini-3.5-transcribe-live": 0.0095,
}
TTS_USD_PER_MIN: Dict[str, float] = {
    "fishaudio/s2.1-pro": 0.0090,
    "cartesia/sonic-3": 0.0300,
    "inworld/inworld-tts-2": 0.0150,
    "deepgram/aura-2": 0.0180,
}
AGENT_SESSION_USD_PER_MIN = 0.0100
# LiveKit's SIP leg only. The carrier leg (Plivo) is billed by Plivo and is
# not included here.
SIP_USD_PER_MIN = 0.0040

_OPENROUTER_TTL_SECS = 6 * 3600
_openrouter_prices: Tuple[float, Dict[str, Tuple[float, float]]] = (0.0, {})
_lock = asyncio.Lock()


def per_minute(table: Dict[str, float], model: str, seconds: float) -> Optional[float]:
    rate = table.get(model)
    return None if rate is None else rate * max(0.0, seconds) / 60.0


async def openrouter_token_prices() -> Dict[str, Tuple[float, float]]:
    """{model: (USD per prompt token, USD per completion token)}, cached 6h.

    List prices from OpenRouter's public model catalog. A fetch failure keeps
    the last good table (or none): cost then reads as unknown, never as zero.
    """
    global _openrouter_prices
    async with _lock:
        fetched_at, prices = _openrouter_prices
        if prices and time.monotonic() - fetched_at < _OPENROUTER_TTL_SECS:
            return prices
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{settings.openrouter_api_base}/models")
                r.raise_for_status()
            fresh: Dict[str, Tuple[float, float]] = {}
            for m in r.json().get("data", []):
                p = m.get("pricing") or {}
                try:
                    fresh[m["id"]] = (float(p.get("prompt") or 0), float(p.get("completion") or 0))
                except (TypeError, ValueError, KeyError):
                    continue
            _openrouter_prices = (time.monotonic(), fresh)
        except Exception:  # noqa: BLE001 - pricing must never break a call
            logger.warning("OpenRouter price list unavailable; LLM cost left unknown", exc_info=True)
        return _openrouter_prices[1]


def llm_cost(prices: Dict[str, Tuple[float, float]], model: str,
             input_tokens: int, output_tokens: int) -> Optional[float]:
    rate = prices.get(model)
    if rate is None:
        return None
    return rate[0] * input_tokens + rate[1] * output_tokens
