"""Authenticated D10 internal tool client."""

import asyncio
import logging
import re
from typing import Any, Dict, List

import httpx

from agent.config import settings

from .context import D10RequestContext

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAYS = (0.25, 1.0)
_TOOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


class D10ToolError(RuntimeError):
    """Raised when D10 rejects a tool request or is unavailable."""


def _headers() -> Dict[str, str]:
    return {
        "x-d10-internal-key": settings.d10_internal_key,
        "Content-Type": "application/json",
    }


async def _request(method: str, url: str, **kwargs: Any) -> httpx.Response:
    """Retry transient transport/5xx responses; never retry caller errors."""
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(method, url, headers=_headers(), **kwargs)
            if response.status_code < 500 and response.status_code != 429:
                return response
            last_error = D10ToolError(
                f"D10 returned transient status {response.status_code}"
            )
        except httpx.HTTPError as exc:
            last_error = exc

        if attempt < _MAX_RETRIES - 1:
            await asyncio.sleep(_RETRY_DELAYS[attempt])

    raise D10ToolError(f"D10 is unavailable after {_MAX_RETRIES} attempts: {last_error}")


def _normalize_tool(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Accept compact D10 declarations and return OpenAI function tools."""
    function = raw.get("function") if raw.get("type") == "function" else raw
    if not isinstance(function, dict):
        raise D10ToolError("D10 tool catalog contained a non-object declaration")
    name = function.get("name")
    if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
        raise D10ToolError("D10 tool catalog contained an invalid tool name")
    parameters = function.get("parameters") or {"type": "object", "properties": {}}
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise D10ToolError(f"D10 tool {name} has an invalid parameter schema")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": str(function.get("description") or "")[:4000],
            "parameters": parameters,
        },
    }


async def fetch_tool_catalog(context: D10RequestContext) -> List[Dict[str, Any]]:
    url = f"{settings.d10_internal_base_url}/internal/agent/tools/catalog"
    response = await _request(
        "GET",
        url,
        params={
            "clinicId": context.clinic_id,
            "userId": context.user_id,
            "actorId": context.actor_id,
            "actorRole": context.actor_role,
            "conversationId": context.conversation_id,
        },
    )
    if response.status_code in (401, 403):
        raise D10ToolError("D10 internal authentication failed")
    if response.status_code >= 400:
        raise D10ToolError(f"D10 tool catalog failed ({response.status_code})")
    try:
        body = response.json()
    except ValueError as exc:
        raise D10ToolError("D10 tool catalog returned invalid JSON") from exc
    raw_tools = body.get("tools") if isinstance(body, dict) else None
    if not isinstance(raw_tools, list):
        raise D10ToolError("D10 tool catalog response is missing tools")
    if len(raw_tools) > 128:
        raise D10ToolError("D10 tool catalog exceeds the 128-tool safety limit")
    return [_normalize_tool(tool) for tool in raw_tools]


async def call_tool(
    *,
    name: str,
    parameters: Dict[str, Any],
    context: D10RequestContext,
    tool_call_id: str,
) -> Dict[str, Any]:
    if not _TOOL_NAME.fullmatch(name):
        raise D10ToolError("The model requested an invalid tool name")
    url = f"{settings.d10_internal_base_url}/internal/agent/tools/{name}"
    response = await _request(
        "POST",
        url,
        json={
            "context": context.tool_envelope(),
            "parameters": parameters,
            "tool_call_id": tool_call_id,
        },
    )
    if response.status_code in (401, 403):
        raise D10ToolError("D10 internal authentication failed")
    try:
        body = response.json()
    except ValueError as exc:
        raise D10ToolError(f"D10 tool {name} returned invalid JSON") from exc
    if response.status_code >= 400:
        detail = body.get("error") if isinstance(body, dict) else None
        raise D10ToolError(f"D10 tool {name} failed ({response.status_code}): {detail or 'unknown error'}")
    if not isinstance(body, dict):
        raise D10ToolError(f"D10 tool {name} returned an invalid response")
    if body.get("success") is False:
        raise D10ToolError(str(body.get("error") or f"D10 tool {name} failed"))
    result = body.get("result", body)
    return result if isinstance(result, dict) else {"value": result}
