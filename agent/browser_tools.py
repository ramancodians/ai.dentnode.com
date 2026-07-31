"""Only the DNLink test-browser tools. Never import Laby tools here."""
from typing import Any, Dict
from google.adk.tools import ToolContext
from .browser_node_client import BrowserToolError, call_browser_tool

def _lab(tool_context: ToolContext) -> str:
    value = (tool_context.state or {}).get("lab_id")
    if not value: raise BrowserToolError("No lab context available")
    return str(value)

async def browser_tabs(tool_context: ToolContext) -> Dict[str, Any]:
    """List the portal tabs connected to the test-only DNLink browser bridge."""
    try: return await call_browser_tool("browser_tabs", _lab(tool_context), {})
    except BrowserToolError as exc: return {"connected": False, "error": str(exc)}

async def browser_command(tool_context: ToolContext, command: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Queue one restricted command for a connected DNLink portal tab.

    Allowed commands: snapshot, navigate, click, type, press_key. First call
    browser_tabs. Navigation is limited to configured portals; there is no
    arbitrary JavaScript, cookies, downloads, uploads, or destructive action.

    Args:
        command: snapshot, navigate, click, type, or press_key.
        args: tabId plus command-specific arguments.
    """
    try: return await call_browser_tool("browser_command", _lab(tool_context), {"command": command, "args": args})
    except BrowserToolError as exc: return {"error": str(exc)}

async def browser_command_result(tool_context: ToolContext, command_id: str) -> Dict[str, Any]:
    """Read the result of a queued DNLink browser command.

    Args:
        command_id: Identifier returned by browser_command.
    """
    try: return await call_browser_tool("browser_command_result", _lab(tool_context), {"command_id": command_id})
    except BrowserToolError as exc: return {"error": str(exc)}

BROWSER_TOOLS = [browser_tabs, browser_command, browser_command_result]