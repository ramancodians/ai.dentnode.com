"""Advisory, bounded tool-family selection; never generates arguments or actions."""
from agent.jev import jev_agent

# Unmapped tools are always retained: extending a backend must not silently hide
# a capability from the orchestrator. Supporting tools remain available too.
FAMILIES = {
    "appointments": ("Check doctor availability or book, reschedule or cancel an appointment", {
        "find_appointment_slots", "book_appointment", "reschedule_appointment", "cancel_appointment", "list_appointments"}),
    "knowledge": ("Clinic fees, location, procedures, payment or insurance information", {"search_clinic_knowledge"}),
    "reception": ("Opening hours, holidays, missed calls or requesting a staff callback", {"get_reception_status", "request_callback", "list_reception_work_items"}),
    "followups": ("Treatment follow-up or appointment confirmation status; prepare a lab staff follow-up task", {"get_patient_follow_up_status", "prepare_follow_up", "task_detail", "staff_list"}),
    "history": ("Read previous calls, call activity, transcripts or summaries", {"get_call_activity", "search_call_conversations", "call_history", "call_summary"}),
    "calling": ("Explicit request to prepare or initiate a patient call or reminder call", {"prepare_patient_call", "patient_reminder_call", "start_background_call"}),
    "records": ("Look up a patient, lab case or patient records", {"search_patients", "get_patient_overview", "get_patient_records", "find_case"}),
}
SUPPORT = {"handoff_to_human", "search_patients", "list_appointments", "staff_list"}


async def select_tools(query, tools):
    names = {tool["function"]["name"] for tool in tools}
    families = [(name, description, members) for name, (description, members) in FAMILIES.items() if names & members]
    if len(families) < 2:
        return tools, "not_needed", None
    # match_text includes an explicit none option plus an independent ambiguity
    # check and conservative confidence threshold. Multiple intents must abstain.
    match = await jev_agent.match_text(
        query, [description for _, description, _ in families], timeout_secs=2.0)
    if match.index is None:
        return tools, "unclear", match
    if not 0 <= match.index < len(families):
        return tools, "unclear", match
    known = set().union(*(members for _, members in FAMILIES.values()))
    selected = families[match.index][2] | SUPPORT | (names - known)
    return [tool for tool in tools if tool["function"]["name"] in selected], "suggested", match
