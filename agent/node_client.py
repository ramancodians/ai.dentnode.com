"""HTTP client for calling back into the Node backend's internal tool API.

The agent never touches the database directly. Every data lookup is a call to
/internal/laby-tools/:tool, which runs the curated Prisma query, scoped to the
lab_id we pass. This keeps the DB query logic in exactly one place (Node).

Transient network/timeout errors are retried up to 3 times with exponential
backoff. Auth (401) and server (4xx/5xx) errors are not retried.
"""

import asyncio
import logging
from typing import Any, Dict

import httpx

from .config import settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAYS = (0.5, 1.5)  # seconds between attempts 1→2 and 2→3


class NodeToolError(RuntimeError):
    pass


async def call_tool(
    tool_name: str,
    lab_id: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Invoke a single curated tool on the Node backend.

    Returns the tool's `result` envelope: {summary, columns, rows, chart_hint, notes}.
    Raises NodeToolError on transport or server failure so the agent can recover.
    """
    url = f"{settings.node_base_url}/internal/laby-tools/{tool_name}"
    payload = {"labId": lab_id, "params": params or {}}
    headers = {
        "x-internal-key": settings.internal_key,
        "Content-Type": "application/json",
    }

    last_exc: Exception | None = None
    resp: httpx.Response | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            last_exc = None
            break
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[attempt]
                logger.warning(
                    "Node tool call failed, retrying",
                    extra={
                        "tool": tool_name,
                        "attempt": attempt + 1,
                        "of": _MAX_RETRIES,
                        "retry_in_secs": delay,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(delay)

    if last_exc is not None:
        logger.error(
            "Node tool unreachable after all retries",
            extra={"tool": tool_name, "attempts": _MAX_RETRIES, "error": str(last_exc)},
        )
        raise NodeToolError(
            f"Could not reach the data service for {tool_name} after {_MAX_RETRIES} attempts: {last_exc}"
        )

    assert resp is not None

    if resp.status_code == 401:
        logger.error("Node internal auth failed", extra={"tool": tool_name})
        raise NodeToolError("Internal authentication failed (check INTERNAL_API_KEY).")

    if resp.status_code >= 400:
        detail = ""
        try:
            detail = resp.json().get("error", "")
        except Exception:  # noqa: BLE001
            detail = resp.text[:200]
        logger.warning(
            "Node tool returned error status",
            extra={"tool": tool_name, "status": resp.status_code, "detail": detail},
        )
        raise NodeToolError(f"{tool_name} failed ({resp.status_code}): {detail}")

    body = resp.json()
    if not body.get("success"):
        error_msg = body.get("error", f"{tool_name} returned no data.")
        logger.warning(
            "Node tool returned failure body",
            extra={"tool": tool_name, "error": error_msg},
        )
        raise NodeToolError(error_msg)

    logger.info("Node tool call succeeded", extra={"tool": tool_name, "lab_id": lab_id})
    return body.get("result", {})


async def fetch_catalog() -> list[dict]:
    """List available tools (used for health/diagnostics, not the hot path)."""
    url = f"{settings.node_base_url}/internal/laby-tools/catalog"
    headers = {"x-internal-key": settings.internal_key}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json().get("tools", [])
