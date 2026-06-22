"""ADK FunctionTools the Laby agent can call.

Design rule: the model chooses WHICH tool and the *business* params (range,
days, top_n…). It never supplies lab_id. We read lab_id from the session state
(`tool_context.state["lab_id"]`), which Node populates from the verified JWT.
This makes cross-tenant data access impossible by construction.

Each tool is a thin async wrapper over the Node internal endpoint; the heavy
lifting (Prisma queries, lab scoping) lives in the Node backend.
"""

from typing import Any, Dict, Optional

from google.adk.tools import ToolContext

from .node_client import NodeToolError, call_tool


def _lab_id(tool_context: ToolContext) -> str:
    lab_id = (tool_context.state or {}).get("lab_id")
    if not lab_id:
        raise NodeToolError("No lab context available for this request.")
    return str(lab_id)


async def _run(tool_context: ToolContext, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return await call_tool(name, _lab_id(tool_context), params)
    except NodeToolError as exc:
        # Returned to the model as a normal result so it can apologise/retry
        # rather than crashing the turn.
        return {
            "summary": f"Could not retrieve this data: {exc}",
            "columns": [],
            "rows": [],
            "chart_hint": None,
            "notes": "tool_error",
        }


async def cases_received(tool_context: ToolContext, range: str = "today") -> Dict[str, Any]:
    """How many NEW cases the lab received, broken down by day.

    Use for: "how many cases did I receive today / this week / this month".

    Args:
        range: One of "today", "yesterday", "this_week", "last_7_days",
            "last_30_days", "this_month". Defaults to "today".
    """
    return await _run(tool_context, "cases_received", {"range": range})


async def cases_timeline(tool_context: ToolContext, weeks: int = 3) -> Dict[str, Any]:
    """Upcoming delivery workload: open cases due over the next N weeks, by date.

    Use for: "what does my timeline look like for the next 3 weeks",
    "what's my upcoming workload".

    Args:
        weeks: Number of weeks ahead to include (1-12). Defaults to 3.
    """
    return await _run(tool_context, "cases_timeline", {"weeks": weeks})


async def expected_volume(tool_context: ToolContext, horizon_days: int = 1) -> Dict[str, Any]:
    """Estimate of NEW cases expected over the next N days (heuristic).

    Use for: "how many cases can I expect tomorrow / this week". This is an
    ESTIMATE based on the recent intake trend — always tell the user so.

    Args:
        horizon_days: Days ahead to estimate (1-14). Defaults to 1 (tomorrow).
    """
    return await _run(tool_context, "expected_volume", {"horizon_days": horizon_days})


async def inactive_clients(
    tool_context: ToolContext, days: int = 30, top_n: int = 25
) -> Dict[str, Any]:
    """Clients (doctors/clinics) who have NOT sent a case in the last N days.

    Use for: "which clients are not sending me cases", "which doctors have
    gone quiet", "who stopped ordering".

    Args:
        days: Inactivity threshold in days. Defaults to 30.
        top_n: Max clients to return, most-inactive first. Defaults to 25.
    """
    return await _run(tool_context, "inactive_clients", {"days": days, "top_n": top_n})


async def product_sales(
    tool_context: ToolContext, range: str = "this_month", top_n: int = 10
) -> Dict[str, Any]:
    """Best-selling products/services by revenue and units, ranked.

    Use for: "which products are selling more", "top products this month".

    Args:
        range: "today", "this_week", "last_7_days", "last_30_days",
            "this_month". Defaults to "this_month".
        top_n: Max products to return. Defaults to 10.
    """
    return await _run(tool_context, "product_sales", {"range": range, "top_n": top_n})


async def staff_activity(tool_context: ToolContext, days: int = 7) -> Dict[str, Any]:
    """Staff who have NOT logged in / been active in the last N days.

    Use for: "which staff are not logging in properly", "who's been inactive".

    Args:
        days: Inactivity threshold in days. Defaults to 7.
    """
    return await _run(tool_context, "staff_activity", {"days": days})


# Registered, in priority order, with the LlmAgent.
LABY_TOOLS = [
    cases_received,
    cases_timeline,
    expected_volume,
    inactive_clients,
    product_sales,
    staff_activity,
]
