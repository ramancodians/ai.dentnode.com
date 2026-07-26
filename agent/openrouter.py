"""Reusable one-shot OpenRouter chat-completion helper.

This is the shared primitive for "single-call" feature agents (insights,
future vision reviewers, classifiers, …) that do NOT need the full ADK tool
loop. It makes exactly one OpenRouter Chat Completions request against the
OpenAI-compatible endpoint and returns a small, uniform result object.

Design goals:
  - Dependency-light: reuses httpx (already in the repo), no OpenAI SDK.
  - Real cost: asks OpenRouter for usage accounting (`usage.include = true`)
    so the response carries the exact credit/USD cost of the call.
  - Vision-ready: `messages` are passed through verbatim, so a caller can put
    OpenAI-style multimodal content (text + image_url parts) in a message now
    or later without any change here.
  - Clean failure: raises OpenRouterError on any HTTP/timeout/transport error
    so the endpoint handler can translate it into an error response and
    status="error" metering.

Usage:
    from agent.openrouter import chat_completion
    result = await chat_completion(
        messages=[{"role": "system", "content": ...},
                  {"role": "user", "content": ...}],
        model=settings.model,          # "openrouter/…" or a bare slug both work
    )
    result.text, result.usage, result.cost_usd, result.latency_ms
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# Generous default; a single completion is not the agent tool loop, but large
# JSON payloads + long reports can still take a while. Callers may override.
_DEFAULT_TIMEOUT_SECS = 60.0


class OpenRouterError(RuntimeError):
    """Raised on any HTTP, timeout, or transport failure talking to OpenRouter."""


@dataclass
class ChatResult:
    text: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: Optional[float] = None
    latency_ms: int = 0


def _strip_route_prefix(model: str) -> str:
    """The native OpenRouter API wants the bare model slug.

    `settings.model` carries the litellm `openrouter/` route prefix (used by the
    ADK agent). This helper talks to OpenRouter directly, so drop that prefix.
    """
    slug = model.strip()
    return slug[len("openrouter/"):] if slug.startswith("openrouter/") else slug


def _extract_usage(raw_usage: Dict[str, Any]) -> Dict[str, int]:
    def _int(v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    return {
        "prompt_tokens": _int(raw_usage.get("prompt_tokens")),
        "completion_tokens": _int(raw_usage.get("completion_tokens")),
        "total_tokens": _int(raw_usage.get("total_tokens")),
    }


async def chat_completion(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout_secs: float = _DEFAULT_TIMEOUT_SECS,
    extra_body: Optional[Dict[str, Any]] = None,
) -> ChatResult:
    """Perform one OpenRouter Chat Completions call and return a ChatResult.

    Args:
        messages: OpenAI-style message list. Content may be a plain string or a
            list of typed parts (text / image_url) for vision-capable models.
        model: OpenRouter model id. Accepts a bare slug ("deepseek/deepseek-v4-flash")
            or the litellm-prefixed "openrouter/…" form; the prefix is stripped.
        temperature: Optional sampling temperature.
        max_tokens: Optional completion cap.
        timeout_secs: Per-request timeout.
        extra_body: Optional additional top-level body fields (merged last;
            e.g. response_format, provider routing prefs).

    Returns:
        ChatResult(text, model, usage, cost_usd, latency_ms).

    Raises:
        OpenRouterError: on missing key, HTTP error status, or transport/timeout.
    """
    if not settings.openrouter_api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not configured")

    slug = _strip_route_prefix(model)

    body: Dict[str, Any] = {
        "model": slug,
        "messages": messages,
        # Ask OpenRouter to include exact usage accounting (cost + native token
        # counts) in the response body.
        "usage": {"include": True},
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if extra_body:
        body.update(extra_body)

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        # Attribution on OpenRouter's dashboard/leaderboards (same as the agent).
        "HTTP-Referer": settings.openrouter_site_url,
        "X-Title": settings.openrouter_app_name,
    }

    url = f"{settings.openrouter_api_base}/chat/completions"
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout_secs) as client:
            resp = await client.post(url, json=body, headers=headers)
    except httpx.HTTPError as exc:
        raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

    latency_ms = int((time.monotonic() - t0) * 1000)

    if resp.status_code >= 400:
        detail = resp.text[:300]
        raise OpenRouterError(
            f"OpenRouter returned {resp.status_code}: {detail}"
        )

    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise OpenRouterError(f"OpenRouter returned non-JSON body: {exc}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise OpenRouterError("OpenRouter response had no choices")

    text = (choices[0].get("message") or {}).get("content") or ""

    raw_usage = data.get("usage") or {}
    usage = _extract_usage(raw_usage)
    # OpenRouter reports the exact cost (in USD credits) under usage.cost when
    # `usage.include` is requested.
    cost_usd: Optional[float] = None
    if raw_usage.get("cost") is not None:
        try:
            cost_usd = float(raw_usage["cost"])
        except (TypeError, ValueError):
            cost_usd = None

    return ChatResult(
        text=text,
        model=data.get("model") or slug,
        usage=usage,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )
