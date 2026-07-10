"""ADK FunctionTools the Laby agent can call.

Design rule: the model chooses WHICH tool and the *business* params (range,
days, top_n…). It never supplies lab_id. We read lab_id from the session state
(`tool_context.state["lab_id"]`), which Node populates from the verified JWT.
This makes cross-tenant data access impossible by construction.

Each tool is a thin async wrapper over the Node internal endpoint; the heavy
lifting (Prisma queries, lab scoping) lives in the Node backend.
"""

from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from .node_client import NodeToolError, call_tool


def _lab_id(tool_context: ToolContext) -> str:
    lab_id = (tool_context.state or {}).get("lab_id")
    if not lab_id:
        raise NodeToolError("No lab context available for this request.")
    return str(lab_id)


def _user_id(tool_context: ToolContext) -> Optional[str]:
    user_id = (tool_context.state or {}).get("user_id")
    return str(user_id) if user_id else None


async def _run(tool_context: ToolContext, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return await call_tool(name, _lab_id(tool_context), params, user_id=_user_id(tool_context))
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

    The result includes an entity_ids field mapping each row to that doctor's
    internal id. You normally don't need it — the action tools take a
    doctor_name — but you may pass a row's id to an action tool as doctor_id
    once you've picked the right doctor. NEVER show these ids to the user.

    Args:
        query: The name / clinic / phone the user mentioned (titles like "dr"
            are fine — they are stripped automatically). Required.
        limit: Max matches to return (1-50). Defaults to 10.
    """
    return await _run(tool_context, "find_doctor", {"query": query, "limit": limit})


async def find_case(tool_context: ToolContext, query: str, limit: int = 20) -> Dict[str, Any]:
    """Search for a case/order by ANY identifier.

    Labs use their own custom IDs — NOT internal Prisma IDs. This tool searches
    ALL known ID fields: case_custom_id, custom_id, order number (#123), book
    ID, scan ID, internal ID, warranty number, and patient name. It also checks
    the lab's configured default_id preference and prioritises accordingly.

    Use for: "find case ABC-123", "search order 456", "look up case by book ID",
    "what is the status of case XYZ", "show me order #500", "find patient
    John's case", "track case with scan ID STL-789".

    The result includes an entity_ids field mapping each row to that case's
    internal id. You may pass a row's id to warranty_create as entry_id once
    you've picked the right case. NEVER show these ids to the user — refer to
    cases by their case_custom_id.

    Args:
        query: The case ID, order number, or patient name to search for.
            Required.
        limit: Max results to return (1-100). Defaults to 20.
    """
    return await _run(tool_context, "find_case", {"query": query, "limit": limit})


async def doctor_financial_analysis(
    tool_context: ToolContext,
    doctor_name: Optional[str] = None,
    doctor_clinic: Optional[str] = None,
    doctor_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Full financial health analysis of a single doctor/client.

    Returns revenue contribution %, outstanding payments with aging buckets
    (0-30, 30-60, 60-90, 90+ days), case volume trend over 3 months, payment
    behavior (collection rate), risk flags, and a 1-5 risk score.

    Use this when the user asks to "analyze" a doctor's finances, assess risk,
    or understand their value to the lab.

    Pass doctor_name — the name the user said (e.g. "Dr. Sharma"); the tool
    resolves the doctor internally. Add doctor_clinic when several doctors share
    a name. If the tool reports the name is ambiguous, ask the user which one
    and call again with the clinic. You do NOT need an internal id — but if a
    prior find_doctor gave you the matching entity_ids value, you may pass it as
    doctor_id instead. Never show that id to the user.

    After calling this, use the domain knowledge in your system prompt to
    interpret the results and recommend actions (collection calls, retention
    moves, referral asks, etc.).

    Args:
        doctor_name: The doctor's name as the user said it. Preferred input.
        doctor_clinic: Optional clinic name to disambiguate same-named doctors.
        doctor_id: Optional internal id from a lookup's entity_ids, if you
            already have it. Never surface it to the user.
    """
    params: Dict[str, Any] = {}
    if doctor_name:
        params["doctor_name"] = doctor_name
    if doctor_clinic:
        params["doctor_clinic"] = doctor_clinic
    if doctor_id:
        params["doctor_id"] = doctor_id
    return await _run(tool_context, "doctor_financial_analysis", params)


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


async def dentnode_guide(tool_context: ToolContext, query: str) -> Dict[str, Any]:
    """Step-by-step guide for using the DentNode product.

    Answers "how do I..." questions: how to create a case, add a doctor,
    generate an invoice, view reports, manage production, configure settings,
    use Laby AI, manage shipments and pickups, and more.

    Pass the user's question as `query`. The tool matches against a curated
    guide database built from real product usage data and returns numbered
    steps with page paths and screenshot references.

    Use this whenever the user asks how to perform a task in DentNode the
    product — not for data queries (use other tools for that).

    Args:
        query: The user's how-to question (e.g. "how do I create a case",
            "how to add a doctor", "how to view financial reports").
    """
    return await _run(tool_context, "dentnode_guide", {"query": query})


async def shipment_summary(
    tool_context: ToolContext,
    range: str = "this_month",
    rollup: str = "month",
    status: Optional[str] = None,
    shipment_type: Optional[str] = None,
    is_e_signed: Optional[bool] = None,
    client_id: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Shipment / dispatch count summary with optional filters.

    Use for: "how many shipments were created today", "how many shipments
    from Jan 1 to Jan 15", "how many shipments are delivered", "how many
    shipments are e-signed", "dispatch report", "monthly dispatch",
    "shipment register", "how many deliveries vs try-ins".

    Returns a count breakdown (by day or month) plus totals for delivered,
    in-transit, pending, e-signed, deliveries, and try-ins.

    Args:
        range: "today", "yesterday", "this_week", "last_7_days",
            "last_30_days", "this_month". Defaults to "this_month".
        rollup: "day" for daily breakdown, "month" for monthly. Defaults
            to "month".
        status: Optional — filter by SHIPMENT_STATUS (PENDING, IN_TRANSIT,
            DELIVERED, CANCELLED).
        shipment_type: Optional — DELIVERY or TRY_IN.
        is_e_signed: Optional — filter to only signed (true) or unsigned
            (false) shipments.
        client_id: Optional — filter to a single doctor/client.
        from_date: Optional — start of explicit date range (YYYY-MM-DD).
            Use with to_date instead of range for arbitrary windows.
        to_date: Optional — end of explicit date range (YYYY-MM-DD).
    """
    params: Dict[str, Any] = {"range": range, "rollup": rollup}
    if status:
        params["status"] = status.upper()
    if shipment_type:
        params["shipment_type"] = shipment_type.upper()
    if is_e_signed is not None:
        params["is_e_signed"] = str(is_e_signed).lower()
    if client_id:
        params["client_id"] = client_id
    if from_date and to_date:
        params["from"] = from_date
        params["to"] = to_date
    return await _run(tool_context, "shipment_summary", params)


async def shipment_list(
    tool_context: ToolContext,
    range: str = "this_month",
    status: Optional[str] = None,
    shipment_type: Optional[str] = None,
    is_e_signed: Optional[bool] = None,
    is_paid: Optional[bool] = None,
    client_id: Optional[str] = None,
    tracking_id: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    """List individual shipments with details and rich filtering.

    Use for: "which shipments are delivered this month", "show me shipments
    signed this week", "list all shipments for Dr. Sharma", "show shipments
    in transit", "which shipments are pending", "shipment register details",
    "find shipment by tracking ID", "show paid shipments", "show my shipments".

    Returns a table with shipment ID, date, client, status, type, delivery
    mode, signed/paid flags, amount, case count, and tracking number.

    Args:
        range: "today", "yesterday", "this_week", "last_7_days",
            "last_30_days", "this_month". Defaults to "this_month".
        status: Optional — PENDING, IN_TRANSIT, DELIVERED, or CANCELLED.
        shipment_type: Optional — DELIVERY or TRY_IN.
        is_e_signed: Optional — filter to signed (true) or unsigned (false).
        is_paid: Optional — filter to paid (true) or unpaid (false).
        client_id: Optional — filter to a single doctor. Use find_doctor
            first if the user only provided a name.
        tracking_id: Optional — partial match on tracking number.
        from_date: Optional — start of explicit range (YYYY-MM-DD). Use
            with to_date instead of range.
        to_date: Optional — end of explicit range (YYYY-MM-DD).
        limit: Max shipments to return (1-200). Defaults to 20.
        offset: Pagination offset (default 0).
    """
    params: Dict[str, Any] = {"range": range, "limit": limit, "offset": offset}
    if status:
        params["status"] = status.upper()
    if shipment_type:
        params["shipment_type"] = shipment_type.upper()
    if is_e_signed is not None:
        params["is_e_signed"] = str(is_e_signed).lower()
    if is_paid is not None:
        params["is_paid"] = str(is_paid).lower()
    if client_id:
        params["client_id"] = client_id
    if tracking_id:
        params["tracking_id"] = tracking_id
    if from_date and to_date:
        params["from"] = from_date
        params["to"] = to_date
    return await _run(tool_context, "shipment_list", params)


async def shipment_detail(
    tool_context: ToolContext, shipment_id: str
) -> Dict[str, Any]:
    """Full detail of a single shipment by ID.

    Use for: "tell me about shipment SHP-123", "show shipment details",
    "what cases are in this shipment", "who signed shipment #25",
    "shipment tracking info", "show me the details of that shipment".

    Returns all shipment fields (status, type, amounts, dates, delivery
    info, e-signature status) plus the list of cases inside it with patient
    names, product types, tooth numbers, and shades. Also includes client
    and staff info.

    Args:
        shipment_id: The shipment's custom_id (e.g. "SHP-001") or internal
            Prisma ID. Required.
    """
    return await _run(tool_context, "shipment_detail", {"shipment_id": shipment_id})


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


async def workflow_summary(tool_context: ToolContext) -> Dict[str, Any]:
    """Overview of all Workflow V2 automations set up in the lab.

    Lists every workflow with its name, manager, covered products, and counts of
    active/pending/completed/failed automation tasks. Use for: "how many
    workflows do I have", "show all workflows", "workflow overview", "list
    automations", "what workflows are set up".

    No parameters needed.
    """
    return await _run(tool_context, "workflow_summary", {})


async def workflow_find_case(
    tool_context: ToolContext, case_id: str
) -> Dict[str, Any]:
    """Find where a specific case/order is in its Workflow V2 automation.

    Tells you: which workflow the case is in, current step name and type
    (e.g. "Milling", "Quality Check"), who is assigned, how many steps are
    completed vs total, whether it's delayed (stalled > 24h), and recent
    task log events showing progress.

    Use for: "where is case ABC-123 in the workflow", "who is working on case
    XYZ", "is case #500 delayed in the workflow", "what step is case DN-001
    on", "show workflow progress for this order", "which department is
    handling case 456".

    Args:
        case_id: The case identifier — case_custom_id, custom_id, or order
            number. Required. Use find_case first if the user only provides
            a patient name.
    """
    return await _run(tool_context, "workflow_find_case", {"case_id": case_id})


async def whatsapp_status(tool_context: ToolContext) -> Dict[str, Any]:
    """Check WhatsApp setup and credit status for the lab.

    Returns: whether a WhatsApp account is linked and active, remaining message
    credits, cost per template message, cost per custom text message, and how
    many messages can be sent with current credits. ALWAYS call this first
    before attempting to send any WhatsApp message.

    Use for: "do I have WhatsApp credits", "check WhatsApp balance",
    "am I linked to WhatsApp", "how many messages can I send".

    No parameters needed.
    """
    return await _run(tool_context, "whatsapp_status", {})


async def whatsapp_templates(tool_context: ToolContext) -> Dict[str, Any]:
    """List all available WhatsApp message templates with their purpose,
    parameter count, and body preview.

    Use this when the user wants to send a message but isn't sure which
    template to use. Show them the options and let them pick. Templates
    cost fewer credits than custom text messages.

    Use for: "what templates are available", "show message templates",
    "what can I send on WhatsApp", "list WhatsApp templates".

    No parameters needed.
    """
    return await _run(tool_context, "whatsapp_templates", {})


async def whatsapp_send(
    tool_context: ToolContext,
    doctor_id: Optional[str] = None,
    recipient: str = "doctor",
    template_name: Optional[str] = None,
    body_params: Optional[List[str]] = None,
    custom_body: Optional[str] = None,
    entry_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a WhatsApp message to a doctor/client, OR to the logged-in user.

    Can send either a pre-built template (payment reminder, case update,
    invoice, etc.) or a custom free-text message. Custom text costs more
    credits. ALWAYS call whatsapp_status first to verify credits, and
    confirm the message content with the user before sending.

    Templates cost 1 credit (linked account) or 3 credits (DentNode
    fallback). Custom text messages cost 1 credit (linked) or 6 credits
    (fallback). Sending to yourself costs the same as sending to a doctor.

    Use for: "send payment reminder to Dr. Sharma", "send case update
    to this doctor", "message Dr. Gupta about his invoice", "send a
    WhatsApp to confirm the case is ready", "send it to me", "send that
    to my number".

    Args:
        doctor_id: The doctor's ID (use find_doctor first). Required when
            recipient is "doctor" (the default). Not needed when recipient
            is "self"/"me".
        recipient: "doctor" (default) to send to the doctor_id above, or
            "self"/"me" to send to the logged-in user's own phone on file.
        template_name: Template name from whatsapp_templates. Optional
            if sending custom_body.
        body_params: List of values for template {{1}}, {{2}}, etc.
            Optional.
        custom_body: Custom text message body. Optional — use instead
            of template_name. Costs more credits.
        entry_id: Optional — link the message to a specific case.
    """
    params: Dict[str, Any] = {"recipient": recipient}
    if doctor_id:
        params["doctor_id"] = doctor_id
    if template_name:
        params["template_name"] = template_name
    if body_params:
        params["body_params"] = body_params
    if custom_body:
        params["custom_body"] = custom_body
    if entry_id:
        params["entry_id"] = entry_id
    return await _run(tool_context, "whatsapp_send", params)


# ----------------------------------------------------------------
# Phase 6 — Actions (draft-first, strictly one action per call)
# ----------------------------------------------------------------


async def shipment_create(
    tool_context: ToolContext,
    doctor_name: Optional[str] = None,
    doctor_clinic: Optional[str] = None,
    doctor_id: Optional[str] = None,
    case_ids: Optional[List[str]] = None,
    delivery_type: str = "IN_HOUSE",
    shipment_type: str = "DELIVERY",
    amount: Optional[int] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    """Create ONE shipment for a SINGLE doctor. DRAFT-FIRST, ONE AT A TIME.

    confirm=false (the default) resolves the doctor's shippable cases and
    the computed amount, and returns a PREVIEW ONLY — it creates nothing.
    Only call again with confirm=true, AFTER showing the draft to the user
    and getting their explicit go-ahead, to actually create the shipment.

    Pass doctor_name — the name the user said; the tool resolves the doctor
    internally (add doctor_clinic to disambiguate same-named doctors). You do
    NOT need an internal id. If the tool reports the name is ambiguous, ask the
    user which doctor and call again with the clinic.

    NEVER create more than one shipment per user request. If the user asks
    for shipments for "all doctors" or "20 shipments", refuse and explain
    you can only create one shipment at a time — ask them to pick one
    doctor.

    Use for: "create a shipment for Dr. Sharma", "ship Dr. Gupta's cases",
    "make a try-in shipment for Dr. X".

    Args:
        doctor_name: The doctor to ship for, as the user named them. Preferred
            input — use find_doctor first only if you need to confirm they
            exist.
        doctor_clinic: Optional clinic name to disambiguate same-named doctors.
        doctor_id: Optional internal id from a lookup's entity_ids, if you
            already have it. Never surface it to the user.
        case_ids: Optional — specific case IDs to include. If omitted, all
            of that doctor's cases ready to ship (not already shipped,
            delivered, or cancelled) are included automatically.
        delivery_type: "IN_HOUSE" or "COURIER". Defaults to "IN_HOUSE".
        shipment_type: "DELIVERY" or "TRY_IN". Defaults to "DELIVERY".
        amount: Optional — override the auto-computed amount.
        confirm: False (default) previews the draft only. True actually
            creates the shipment. NEVER set true without first showing the
            draft and getting the user's explicit confirmation.
    """
    params: Dict[str, Any] = {
        "delivery_type": delivery_type,
        "shipment_type": shipment_type,
        "confirm": confirm,
    }
    if doctor_name:
        params["doctor_name"] = doctor_name
    if doctor_clinic:
        params["doctor_clinic"] = doctor_clinic
    if doctor_id:
        params["doctor_id"] = doctor_id
    if case_ids:
        params["case_ids"] = case_ids
    if amount is not None:
        params["amount"] = amount
    return await _run(tool_context, "shipment_create", params)


async def warranty_create(
    tool_context: ToolContext,
    entry_id: Optional[str] = None,
    doctor_name: Optional[str] = None,
    doctor_clinic: Optional[str] = None,
    doctor_id: Optional[str] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    """Create ONE warranty card (with a real PDF) for a SINGLE case.
    DRAFT-FIRST, ONE AT A TIME.

    confirm=false (the default) resolves the case and returns a PREVIEW of
    the patient, doctor, work, warranty number, and valid-until date — it
    generates NOTHING. Only call again with confirm=true, AFTER showing the
    draft and getting the user's explicit go-ahead, to actually render and
    store the PDF.

    On confirm=true this produces a downloadable PDF. Tell the user
    afterwards that they can download it or send it to the doctor / to
    themselves on WhatsApp.

    NEVER create more than one warranty card per user request.

    Use for: "create a warranty card for case 1234", "generate a warranty
    card for Dr. Sharma's newest case", "make an e-warranty card".

    Args:
        entry_id: The case to generate a warranty card for. Use the lab's
            case ID (case_custom_id) from find_case, or the internal id from
            find_case's entity_ids. Provide this OR a doctor.
        doctor_name: If entry_id is omitted, the doctor's name as the user
            said it — resolves to that doctor's NEWEST case. Add doctor_clinic
            to disambiguate same-named doctors.
        doctor_clinic: Optional clinic name to disambiguate same-named doctors.
        doctor_id: Optional internal id from a lookup's entity_ids, if you
            already have it. Never surface it to the user.
        confirm: False (default) previews the draft only. True actually
            renders and stores the PDF. NEVER set true without first
            showing the draft and getting the user's explicit confirmation.
    """
    params: Dict[str, Any] = {"confirm": confirm}
    if entry_id:
        params["entry_id"] = entry_id
    if doctor_name:
        params["doctor_name"] = doctor_name
    if doctor_clinic:
        params["doctor_clinic"] = doctor_clinic
    if doctor_id:
        params["doctor_id"] = doctor_id
    return await _run(tool_context, "warranty_create", params)


async def staff_list(
    tool_context: ToolContext, query: str = "", limit: int = 25
) -> Dict[str, Any]:
    """List active staff members in the lab.

    Use for: "who works here", "list my staff", "find Rahul", "who can I
    assign this task to", "show me all staff".

    This is the FIRST tool to call when the user wants to create a task
    and needs to pick an assignee — it returns names AND internal ids
    (in entity_ids) that task_create needs.

    Args:
        query: Optional name filter. Case-insensitive partial match.
        limit: Max results (1-100). Defaults to 25.
    """
    return await _run(tool_context, "staff_list", {"query": query, "limit": limit})


async def task_list(
    tool_context: ToolContext,
    scope: str = "assigned",
    status: Optional[str] = None,
    assigned_to_id: Optional[str] = None,
) -> Dict[str, Any]:
    """List tasks/to-dos in the lab with their status.

    Use for: "what tasks are assigned to me", "show all open tasks",
    "what tasks does Rahul have", "show my pending work", "tasks
    created by me", "how many tasks are done".

    Args:
        scope: "assigned" (tasks assigned to current user, default),
               "created" (tasks created by current user),
               "all" (all tasks in the lab).
        status: Optional filter — "OPEN", "COMPLETED", or "REJECTED".
        assigned_to_id: Optional internal staff id from staff_list
            entity_ids to see a specific person's tasks.
    """
    params: Dict[str, Any] = {"scope": scope}
    if status:
        params["status"] = status
    if assigned_to_id:
        params["assigned_to_id"] = assigned_to_id
    return await _run(tool_context, "task_list", params)


async def task_create(
    tool_context: ToolContext,
    description: str,
    due_date: str,
    assigned_to_id: str,
    image_url: Optional[str] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    """Create ONE task/to-do and assign it to a staff member. DRAFT-FIRST.

    confirm=false (the default) returns a PREVIEW of what will be created
    — it creates NOTHING. Only call again with confirm=true AFTER showing
    the draft and getting the user's explicit go-ahead.

    NEVER create more than one task per call. NEVER call with confirm=true
    without first showing the draft and getting user confirmation.

    Use for: "create a task for Rahul to call the courier", "assign a
    to-do to Priya", "remind someone to do X by Friday".

    Args:
        description: The one-line task instruction (required).
        due_date: ISO date string like "2026-07-15" (required).
        assigned_to_id: Internal staff id from staff_list entity_ids
            (required — use staff_list first to find the person).
        image_url: Optional S3 URL for an attached image.
        confirm: False (default) previews the draft only. True actually
            creates the task and notifies the assignee.
    """
    params: Dict[str, Any] = {
        "description": description,
        "due_date": due_date,
        "assigned_to_id": assigned_to_id,
        "confirm": confirm,
    }
    if image_url:
        params["image_url"] = image_url
    return await _run(tool_context, "task_create", params)


async def task_detail(
    tool_context: ToolContext, task_id: str
) -> Dict[str, Any]:
    """Get full detail of a single task by its ID.

    Use after task_list when the user wants more info on a specific task
    or asks "tell me more about that task".

    Args:
        task_id: The task's internal id from task_list entity_ids.
    """
    return await _run(tool_context, "task_detail", {"task_id": task_id})


async def task_complete(
    tool_context: ToolContext, task_id: str
) -> Dict[str, Any]:
    """Mark a task as COMPLETED.

    Only the assignee or an admin can complete a task. If the caller
    is neither, the tool returns an error.

    Use for: "mark that task as done", "complete the courier task",
    "I finished that to-do".

    Args:
        task_id: The task's internal id from task_list entity_ids.
    """
    return await _run(tool_context, "task_complete", {"task_id": task_id})


async def task_reject(
    tool_context: ToolContext, task_id: str, reason: str
) -> Dict[str, Any]:
    """Reject a task with a reason. Only the assignee or an admin can reject.

    The reason is REQUIRED — ask the user why they want to reject if
    they haven't said.

    Use for: "reject that task", "I can't do this", "decline the to-do".

    Args:
        task_id: The task's internal id from task_list entity_ids.
        reason: Why the task is being rejected (required).
    """
    return await _run(tool_context, "task_reject", {"task_id": task_id, "reason": reason})


async def task_notify(
    tool_context: ToolContext,
    task_id: str,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """Send an in-app notification about a task.

    Use to remind an assignee about a pending task, or notify the
    task creator about a status change.

    Use for: "remind Rahul about the report task", "notify Priya
    about the pending courier task", "ping them about this".

    Args:
        task_id: The task's internal id from task_list entity_ids.
        message: Optional custom notification message. Defaults to
            a reminder about the due date or status.
    """
    params: Dict[str, Any] = {"task_id": task_id}
    if message:
        params["message"] = message
    return await _run(tool_context, "task_notify", params)


# Registered, in priority order, with the LlmAgent.
LABY_TOOLS = [
    # Meta / navigation
    report_directory,
    dentnode_guide,
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
    find_case,
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
    doctor_financial_analysis,
    # Phase 3 — report-aligned tools
    shipment_summary,
    shipment_list,
    shipment_detail,
    expense_summary,
    payment_mode_breakdown,
    daily_order_activity,
    city_wise_orders,
    pickup_summary,
    stock_summary,
    technician_activity,
    # Phase 4 — workflow automation
    workflow_summary,
    workflow_find_case,
    # Phase 5 — WhatsApp messaging
    whatsapp_status,
    whatsapp_templates,
    whatsapp_send,
    # Phase 6 — Actions (draft-first, one action at a time)
    shipment_create,
    warranty_create,
    # Phase 7 — Tasks (to-do / ticket system)
    staff_list,
    task_list,
    task_create,
    task_detail,
    task_complete,
    task_reject,
    task_notify,
]
