"""Central AI usage/metering reporter.

Every AI/LLM call in the platform should be metered into exactly one place: the
Node backend's AiUsageEvent ledger, via POST /internal/ai-usage. This module is
the single client for that endpoint.

CRITICAL: metering is best-effort and MUST NOT affect the actual AI response.
`report_usage` never raises and never blocks the hot path — any failure (network,
auth, bad payload, Node down) is swallowed and logged at WARNING. Callers should
fire-and-forget it (e.g. `asyncio.create_task(report_usage(...))`) or await it
inside their own try/except; either way it will not propagate an error.

The endpoint lives under the same /api prefix as the laby-tools endpoints, so we
reuse `settings.node_base_url` (which already includes /api) and the same
x-internal-key auth header used by node_client.py.
"""

import logging
from typing import Any, Dict, Optional

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# Best-effort estimate used only when OpenRouter's exact cost is unavailable.
# Kept deliberately rough — the source of truth for cost is OpenRouter; this is
# a placeholder so dashboards are never blank. Feature agents may override the
# passed-in `cost` with a better figure when they have one.
_ESTIMATED_USD_PER_1K_TOKENS = 0.0

# Short timeout: metering must never delay the response. If Node is slow we drop
# the event rather than hold the request.
_TIMEOUT_SECS = 5.0


def _coerce_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


async def report_usage(
    *,
    feature: str,
    lab_id: str,
    model: str,
    usage: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    cost: Optional[float] = None,
    cost_source: str = "estimated",
    latency_ms: Optional[int] = None,
    status: str = "ok",
    request_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Record one AI usage event in the Node ledger. Never raises.

    Args:
        feature: Logical feature name, e.g. "laby_chat", "scan_review".
        lab_id:  Trusted lab id for the call (never model-supplied).
        model:   Model id used (OpenRouter slug or litellm route).
        usage:   Token usage dict. Accepts either OpenRouter/OpenAI keys
                 (prompt_tokens / completion_tokens / total_tokens) or ADK/GenAI
                 keys (prompt_token_count / candidates_token_count /
                 total_token_count).
        user_id: Trusted user id, if known.
        cost:    Cost in USD. If None, falls back to a best-effort estimate.
        cost_source: "openrouter" (exact) or "estimated".
        latency_ms:  Wall-clock latency of the call, if measured.
        status:  "ok" or "error".
        request_id:  Correlation id, if any.
        meta:    Arbitrary JSON-serialisable extra context.
    """
    try:
        usage = usage or {}
        prompt_tokens = _coerce_int(
            usage.get("prompt_tokens", usage.get("prompt_token_count"))
        )
        completion_tokens = _coerce_int(
            usage.get("completion_tokens", usage.get("candidates_token_count"))
        )
        total_tokens = _coerce_int(
            usage.get("total_tokens", usage.get("total_token_count"))
        )
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens

        if cost is None:
            cost = round(
                (total_tokens / 1000.0) * _ESTIMATED_USD_PER_1K_TOKENS, 6
            )
            cost_source = "estimated"

        payload: Dict[str, Any] = {
            "feature": feature,
            "lab_id": lab_id,
            "user_id": user_id,
            "provider": "openrouter",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost,
            "cost_source": cost_source,
            "latency_ms": latency_ms,
            "status": status,
            "request_id": request_id,
            "meta": meta,
        }

        url = f"{settings.node_base_url}/internal/ai-usage"
        headers = {
            "x-internal-key": settings.internal_key,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT_SECS) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code >= 400:
            logger.warning(
                "AI usage report rejected by Node",
                extra={
                    "feature": feature,
                    "lab_id": lab_id,
                    "status": resp.status_code,
                },
            )
    except Exception as exc:  # noqa: BLE001 - metering must never break the caller
        logger.warning(
            "AI usage report failed (non-fatal)",
            extra={"feature": feature, "lab_id": lab_id, "error": str(exc)},
        )
