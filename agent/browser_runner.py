"""Runner for the isolated DNLink browser controller."""
import asyncio
import logging
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from .config import settings
from .browser_agent import browser_agent

logger = logging.getLogger(__name__)
_sessions = InMemorySessionService()
_runner = Runner(app_name="dnlink_browser", agent=browser_agent, session_service=_sessions)

async def run_browser_turn(*, lab_id: str, user_id: str, question: str, history: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[Dict[str, Any], None]:
    session = await _sessions.create_session(app_name="dnlink_browser", user_id=user_id, state={"lab_id": lab_id, "user_id": user_id})
    prior = "\n".join(f"{t.get('role', 'user')}: {t.get('text', '')}" for t in (history or [])[-6:])
    message = genai_types.Content(role="user", parts=[genai_types.Part(text=f"{prior}\nUser: {question}" if prior else question)])
    yield {"type": "status", "step": "thinking"}
    try:
        async with asyncio.timeout(settings.turn_timeout_secs):
            async for event in _runner.run_async(user_id=user_id, session_id=session.id, new_message=message):
                for part in list(getattr(getattr(event, "content", None), "parts", None) or []):
                    call = getattr(part, "function_call", None)
                    if call is not None:
                        yield {"type": "status", "step": "calling_tool"}
                        yield {"type": "tool_call", "name": getattr(call, "name", "unknown"), "params": dict(getattr(call, "args", {}) or {})}
                        continue
                    response = getattr(part, "function_response", None)
                    if response is not None:
                        yield {"type": "tool_result", "name": getattr(response, "name", "unknown"), "result": getattr(response, "response", {}) or {}}
                        continue
                    text = getattr(part, "text", None)
                    if text:
                        yield {"type": "status", "step": "responding"}
                        yield {"type": "delta", "text": text}
        yield {"type": "done"}
    except asyncio.TimeoutError:
        yield {"type": "error", "code": "TURN_TIMEOUT", "message": "Browser controller timed out."}
    except Exception:
        yield {"type": "error", "code": "BROWSER_AGENT_FAILED", "message": "Browser controller could not complete that request."}