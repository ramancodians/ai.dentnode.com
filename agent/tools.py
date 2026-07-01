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


# ----------------------------------------------------------------
# Phase 1 — core counts & lists
# ----------------------------------------------------------------


async def lab_overview(tool_context: ToolContext) -> Dict[str, Any]:
    """Dashboard snapshot: total cases, doctors, patients, staff, revenue.

    Use for: "tell me about my lab", "lab overview", "how many total cases",
    "how many total doctors", "how many patients", "how many staff".
    No parameters needed — returns all-time counts.
    """
    return await _run(tool_context, "lab_overview", {})


async def doctor_list(
    tool_context: ToolContext, top_n: int = 20, sort_by: str = "cases"
) -> Dict[str, Any]:
    """Full doctor/client directory sorted by case count, revenue, or newest.

    Use for: "how many doctors", "who are my top doctors", "list all doctors",
    "show me all clients", "which doctor sends the most cases".

    Args:
        top_n: Max doctors to return (1-200). Defaults to 20.
        sort_by: "cases", "revenue", or "newest". Defaults to "cases".
    """
    return await _run(tool_context, "doctor_list", {"top_n": top_n, "sort_by": sort_by})


async def find_doctor(tool_context: ToolContext, query: str, limit: int = 10) -> Dict[str, Any]:
    """Look up a SPECIFIC doctor/client by name, clinic, phone, or email.

    ALWAYS use this when the user names a doctor or asks about ONE person —
    e.g. "tell me about dr. raman", "how is doctor sharma doing", "find Apex
    Dental", "what does raman owe me". It searches the whole directory (not just
    the top doctors) and returns that doctor's case count, revenue, outstanding
    balance and last case date. Do NOT use `doctor_list` for a named doctor —
    that only returns the top few and will miss anyone outside it.

    If several people match, the result lists them so you can ask the user which
    one they mean. If nothing matches, say so and offer to list all doctors.

    Args:
        query: The name / clinic / phone the user mentioned (titles like "dr"
            are fine — they are stripped automatically). Required.
        limit: Max matches to return (1-50). Defaults to 10.
    """
    return await _run(tool_context, "find_doctor", {"query": query, "limit": limit})


async def case_status_breakdown(
    tool_context: ToolContext, range: str = "this_month"
) -> Dict[str, Any]:
    """Distribution of cases by status in a date range.

    Use for: "how many cases are pending", "how many in progress",
    "how many completed", "case status summary".

    Args:
        range: "today", "this_week", "last_7_days", "last_30_days",
            "this_month". Defaults to "this_month".
    """
    return await _run(tool_context, "case_status_breakdown", {"range": range})


async def delayed_cases(tool_context: ToolContext, top_n: int = 20) -> Dict[str, Any]:
    """Cases past their expected delivery date that are still open.

    Use for: "show me delayed cases", "which cases are late", "overdue cases".

    Args:
        top_n: Max cases to return. Defaults to 20.
    """
    return await _run(tool_context, "delayed_cases", {"top_n": top_n})


async def outstanding_payments(
    tool_context: ToolContext,
    top_n: int = 20,
    doctor_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Doctors with outstanding (unpaid) invoice balances, ranked highest first.

    Use for: "who owes me money", "show unpaid invoices", "outstanding amounts",
    "which doctor has pending payments".

    Args:
        top_n: Max doctors to return (1-200). Defaults to 20.
        doctor_id: Optional — filter to a single doctor.
    """
    params: Dict[str, Any] = {"top_n": top_n}
    if doctor_id:
        params["doctor_id"] = doctor_id
    return await _run(tool_context, "outstanding_payments", params)


async def patient_count(tool_context: ToolContext) -> Dict[str, Any]:
    """Total patient count with active (case in last 90 days) and new-this-month
    breakdown.

    Use for: "how many patients do I have", "patient count".
    No parameters needed.
    """
    return await _run(tool_context, "patient_count", {})


# ----------------------------------------------------------------
# Phase 2 — analytics & comparisons
# ----------------------------------------------------------------


async def revenue_summary(
    tool_context: ToolContext, range: str = "this_month"
) -> Dict[str, Any]:
    """Revenue summary: billed, collected, outstanding, invoices for a period.

    Use for: "what is my total revenue", "revenue this month", "how much did I
    bill", "how much did I collect".

    Args:
        range: "today", "this_week", "last_7_days", "last_30_days",
            "this_month". Defaults to "this_month".
    """
    return await _run(tool_context, "revenue_summary", {"range": range})


async def month_over_month(
    tool_context: ToolContext, metric: str = "cases"
) -> Dict[str, Any]:
    """Compare this month vs last month for a chosen metric.

    Use for: "compare this month to last month", "is my business growing",
    "month over month", "trend".

    Args:
        metric: "cases", "revenue", or "collections". Defaults to "cases".
    """
    return await _run(tool_context, "month_over_month", {"metric": metric})


async def turnaround_time(
    tool_context: ToolContext,
    range: str = "last_30_days",
) -> Dict[str, Any]:
    """Average/median/min/max turnaround time (days from order to completion).

    Use for: "average turnaround time", "how fast am I delivering", "TAT".

    Args:
        range: "today", "this_week", "last_7_days", "last_30_days",
            "this_month". Defaults to "last_30_days".
    """
    return await _run(tool_context, "turnaround_time", {"range": range})


async def busy_day_analysis(
    tool_context: ToolContext, days: int = 90
) -> Dict[str, Any]:
    """Which days of the week get the most cases.

    Use for: "which is my busiest day", "what day do most orders come in",
    "weekly pattern".

    Args:
        days: Lookback window in days (7-365). Defaults to 90.
    """
    return await _run(tool_context, "busy_day_analysis", {"days": days})


async def rejection_rate(
    tool_context: ToolContext,
    range: str = "this_month",
) -> Dict[str, Any]:
    """Rejection/repeat rate: cancelled or repeated vs total cases.

    Use for: "what is my rejection rate", "quality rate".

    Args:
        range: "today", "this_week", "last_7_days", "last_30_days",
            "this_month". Defaults to "this_month".
    """
    return await _run(tool_context, "rejection_rate", {"range": range})


# ----------------------------------------------------------------
# Phase 3 — report-aligned tools
# ----------------------------------------------------------------


async def report_directory(tool_context: ToolContext) -> Dict[str, Any]:
    """List ALL available reports in DentNode, grouped by category.

    Use for: "what reports do you have", "show me all reports", "which reports
    are available", "what can I see in reports", "help me find a report".
    No parameters needed.
    """
    return await _run(tool_context, "report_directory", {})


async def shipment_summary(
    tool_context: ToolContext,
    range: str = "this_month",
    rollup: str = "month",
) -> Dict[str, Any]:
    """Shipment / dispatch summary: how many shipments were sent in a period.

    Use for: "how many shipments did I send", "dispatch report", "monthly
    dispatch", "shipment register", "how many cases were delivered".
    Relates to the Shipment Register and Month-wise Dispatch reports.

    Args:
        range: "today", "this_week", "last_7_days", "last_30_days",
            "this_month". Defaults to "this_month".
        rollup: "day" for daily breakdown, "month" for monthly. Defaults to
            "month".
    """
    return await _run(tool_context, "shipment_summary", {"range": range, "rollup": rollup})


async def expense_summary(
    tool_context: ToolContext,
    range: str = "this_month",
    rollup: str = "month",
) -> Dict[str, Any]:
    """Expense summary: total amount spent in a period, by day or month.

    Use for: "what are my expenses", "how much did I spend", "expense report",
    "monthly expenses", "expense register".
    Relates to the Expenses List and Monthly Expenses reports.

    Args:
        range: "today", "this_week", "last_7_days", "last_30_days",
            "this_month". Defaults to "this_month".
        rollup: "day" or "month". Defaults to "month".
    """
    return await _run(tool_context, "expense_summary", {"range": range, "rollup": rollup})


async def payment_mode_breakdown(
    tool_context: ToolContext,
    range: str = "this_month",
) -> Dict[str, Any]:
    """Payments collected broken down by mode: Cash, UPI, Card, Cheque, etc.

    Use for: "how much cash did I collect", "payment mode summary", "UPI vs
    cash collections", "how are payments coming in".
    Relates to the Payment Mode Summary report.

    Args:
        range: "today", "this_week", "last_7_days", "last_30_days",
            "this_month". Defaults to "this_month".
    """
    return await _run(tool_context, "payment_mode_breakdown", {"range": range})


async def daily_order_activity(
    tool_context: ToolContext,
    range: str = "last_7_days",
) -> Dict[str, Any]:
    """Day-wise order creation count and unique client count.

    Use for: "how many orders per day", "daily order activity", "daily order
    diary", "orders created each day", "day wise production".
    Relates to the Daily Order Activity report.

    Args:
        range: "this_week", "last_7_days", "this_month", "last_30_days".
            Defaults to "last_7_days".
    """
    return await _run(tool_context, "daily_order_activity", {"range": range})


async def city_wise_orders(
    tool_context: ToolContext,
    range: str = "this_month",
) -> Dict[str, Any]:
    """Orders grouped by client city: city, client count, order count.

    Use for: "which cities are ordering the most", "city wise order summary",
    "orders by city", "which city sends the most work".
    Relates to the City-wise Order Summary and City-wise Clients reports.

    Args:
        range: "this_month", "last_30_days", "this_week", "last_7_days",
            "this_year". Defaults to "this_month".
    """
    return await _run(tool_context, "city_wise_orders", {"range": range})


async def pickup_summary(
    tool_context: ToolContext,
    range: str = "this_month",
) -> Dict[str, Any]:
    """Pickup requests: total scheduled, fulfilled, pending, and per-staff count.

    Use for: "show pickup schedule", "how many pickups are done", "pickup
    report", "which staff did the most pickups", "pending pickups".
    Relates to the Scheduled Pickup Requests and Staff-wise Pickups Done reports.

    Args:
        range: "today", "this_week", "this_month", "last_30_days". Defaults to
            "this_month".
    """
    return await _run(tool_context, "pickup_summary", {"range": range})


async def stock_summary(tool_context: ToolContext) -> Dict[str, Any]:
    """Current inventory levels for all stock items in the lab.

    Use for: "show stock levels", "stock summary", "what stock do I have",
    "inventory report".
    Relates to the Stock Summary and Stock Requests reports.
    No parameters needed.
    """
    return await _run(tool_context, "stock_summary", {})


async def technician_activity(
    tool_context: ToolContext,
    range: str = "this_month",
) -> Dict[str, Any]:
    """Jobs assigned to each technician/staff: job count and units, ranked.

    Use for: "technician activity log", "staff job report", "how many jobs did
    each technician do", "technician performance", "who is doing the most work".
    Relates to the Technician Activity Log and Technician Jobs reports.

    Args:
        range: "this_month", "last_30_days", "this_week", "last_7_days".
            Defaults to "this_month".
    """
    return await _run(tool_context, "technician_activity", {"range": range})


# Registered, in priority order, with the LlmAgent.
LABY_TOOLS = [
    # Meta / navigation
    report_directory,
    # Phase 0 (original)
    cases_received,
    cases_timeline,
    expected_volume,
    inactive_clients,
    product_sales,
    staff_activity,
    # Phase 1 — core counts & lists
    lab_overview,
    doctor_list,
    find_doctor,
    case_status_breakdown,
    delayed_cases,
    outstanding_payments,
    patient_count,
    # Phase 2 — analytics & comparisons
    revenue_summary,
    month_over_month,
    turnaround_time,
    busy_day_analysis,
    rejection_rate,
    # Phase 3 — report-aligned tools
    shipment_summary,
    expense_summary,
    payment_mode_breakdown,
    daily_order_activity,
    city_wise_orders,
    pickup_summary,
    stock_summary,
    technician_activity,
]
