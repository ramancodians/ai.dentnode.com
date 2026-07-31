"""Client for the isolated DNLink test-browser tool proxy."""
from typing import Any, Dict
import httpx
from .config import settings

class BrowserToolError(RuntimeError): pass

async def call_browser_tool(tool_name: str, lab_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{settings.node_base_url}/internal/browser-tools/{tool_name}"
    headers = {"x-internal-key": settings.internal_key, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json={"labId": lab_id, "params": params or {}}, headers=headers)
    except httpx.HTTPError as exc:
        raise BrowserToolError(f"Browser bridge unreachable: {exc}") from exc
    if response.status_code >= 400:
        try: detail = response.json().get("error", "Browser bridge failed")
        except Exception: detail = response.text[:200]
        raise BrowserToolError(str(detail))
    body = response.json()
    if not body.get("success"): raise BrowserToolError(str(body.get("error", "Browser bridge failed")))
    return body.get("result", {})