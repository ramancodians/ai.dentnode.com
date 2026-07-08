"""Run the Laby agent for one user turn and yield normalized stream events.

The Node backend owns durable chat history (the Laby* Prisma tables). It passes
us the recent turns as `history`, plus the verified `lab_id`. We build a fresh
ADK session per request, seed it with that context, run the agent, and stream
back a small, stable event vocabulary that Node translates into its SSE
protocol for the browser.

Event vocabulary emitted (each is a dict):
  {"type": "status", "step": "thinking"|"calling_tool"|"responding"}
  {"type": "tool_call", "name": str, "params": dict}
  {"type": "tool_result", "name": str, "summary": str,
        "columns": [...], "rows": [...], "chart_hint": {...}|None, "notes": str|None,
        "export": {"url": str, "filename": str}|None}
  {"type": "delta", "text": str}
  {"type": "done"}
  {"type": "error", "code": str, "message": str}
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from .config import settings
from .root_agent import root_agent

logger = logging.getLogger(__name__)
APP_NAME = "laby"

# One in-process session service. Sessions are ephemeral (one per request);
# durable memory lives in Node, so scaling to zero on Cloud Run is safe.
_session_service = InMemorySessionService()
_runner = Runner(
    app_name=APP_NAME,
    agent=root_agent,
    session_service=_session_service,
)


def _history_to_preamble(history: List[Dict[str, str]]) -> str:
    """Render recent turns as a compact text preamble for short-term memory."""
    if not history:
        return ""
    lines = ["Recent conversation (most recent last):"]
    for turn in history[-settings.history_turns:]:
        role = turn.get("role", "user").lower()
        speaker = "User" if role in ("user", "human") else "Laby"
        text = (turn.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _mentions_to_context(mentions: List[Dict[str, str]]) -> str:
    """Render @mentions as context for the agent.

    IMPORTANT: never put the internal person id in the prompt text — the model
    tends to parrot it back to the user (e.g. "ID: clwqhjk..."). We surface only
    the name and role here; the id (if ever needed) lives in session state.
    """
    if not mentions:
        return ""
    lines = ["The user has tagged the following people with @:"]
    for m in mentions:
        name = m.get("name", "Unknown")
        role = m.get("role", "unknown")
        lines.append(f"  - {name} (role: {role})")
    lines.append(
        "When the user asks about these people, use find_doctor for doctors "
        "or the relevant tool for staff. Answer specifically about them."
    )
    return "\n".join(lines)


def _mentions_to_state(mentions: List[Dict[str, str]]) -> Dict[str, str]:
    """Map tagged people's names to their internal ids for session state.

    Kept OUT of the prompt (see _mentions_to_context) so the id never leaks into
    model-visible text, while still being available to tools if needed later.
    """
    state: Dict[str, str] = {}
    for m in mentions:
        name = (m.get("name") or "").strip().lower()
        person_id = (m.get("id") or "").strip()
        if name and person_id:
            state[name] = person_id
    return state


async def run_turn(
    *,
    lab_id: str,
    user_id: str,
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    mentions: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Execute one turn; yields normalized event dicts."""
    # Seed a fresh session with lab_id in state (tools read it from there) and
    # the recent history as initial context.
    session_state: Dict[str, Any] = {"lab_id": lab_id, "user_id": user_id}
    mention_state = _mentions_to_state(mentions or [])
    if mention_state:
        # Tagged people's ids, keyed by lowercased name — available to tools,
        # never rendered into the prompt (see _mentions_to_context).
        session_state["mentioned_people"] = mention_state

    session = await _session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        state=session_state,
    )

    preamble = _history_to_preamble(history or [])
    mention_ctx = _mentions_to_context(mentions or [])
    parts = [p for p in [preamble, mention_ctx] if p]
    context = "\n\n".join(parts) if parts else ""
    prompt = f"{context}\n\nUser: {question}" if context else question
    new_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=prompt)],
    )

    logger.info("Agent turn started", extra={"lab_id": lab_id, "user_id": user_id})
    yield {"type": "status", "step": "thinking"}

    try:
        async with asyncio.timeout(settings.turn_timeout_secs):
            async for event in _runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=new_message,
            ):
                # Surface function calls / responses as they happen.
                for part in _parts(event):
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        tool_name = getattr(fc, "name", "unknown")
                        logger.info(
                            "Agent calling tool",
                            extra={"tool": tool_name, "lab_id": lab_id},
                        )
                        yield {"type": "status", "step": "calling_tool"}
                        yield {
                            "type": "tool_call",
                            "name": tool_name,
                            "params": dict(getattr(fc, "args", {}) or {}),
                        }
                        continue

                    fr = getattr(part, "function_response", None)
                    if fr is not None:
                        payload = getattr(fr, "response", {}) or {}
                        if isinstance(payload, dict):
                            yield {
                                "type": "tool_result",
                                "name": getattr(fr, "name", "unknown"),
                                "summary": payload.get("summary", ""),
                                "columns": payload.get("columns", []),
                                "rows": payload.get("rows", []),
                                "chart_hint": payload.get("chart_hint"),
                                "notes": payload.get("notes"),
                                "export": payload.get("export"),
                                "entity_ids": payload.get("entity_ids"),
                            }
                        continue

                    text = getattr(part, "text", None)
                    if text:
                        yield {"type": "status", "step": "responding"}
                        yield {"type": "delta", "text": text}

        logger.info("Agent turn completed", extra={"lab_id": lab_id})
        yield {"type": "done"}

    except asyncio.TimeoutError:
        logger.error(
            "Agent turn timed out",
            extra={"lab_id": lab_id, "timeout_secs": settings.turn_timeout_secs},
        )
        yield {
            "type": "error",
            "code": "TURN_TIMEOUT",
            "message": "Laby took too long to respond. Please try again.",
        }
    except Exception as exc:  # noqa: BLE001 - report any agent failure cleanly
        logger.error(
            "Agent turn failed",
            extra={"lab_id": lab_id, "error": str(exc)},
            exc_info=True,
        )
        yield {
            "type": "error",
            "code": "AGENT_FAILED",
            "message": f"Laby could not complete that request: {exc}",
        }


def _parts(event: Any) -> list:
    content = getattr(event, "content", None)
    if content is None:
        return []
    return list(getattr(content, "parts", None) or [])
