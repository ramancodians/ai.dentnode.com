"""Advisory Jev matching for tenant-scoped search results.

The data services still own filtering and authorization. This module sends only
the query and short candidate labels to Jev, and never selects an entity for an
action or removes an ambiguous result.
"""

from __future__ import annotations

from typing import Any

from agent.jev import TextMatchResult, jev_agent


def _candidate_labels(tool_name: str, query: str, result: dict[str, Any]) -> list[str]:
    if tool_name in {"search_patients", "prepare_patient_call"}:
        key = "patients" if tool_name == "search_patients" else "candidates"
        patients = result.get(key)
        if not isinstance(patients, list):
            return []
        if any(not isinstance(patient, dict) or not isinstance(patient.get("name"), str) for patient in patients):
            return []
        # IDs and contact details are exact database lookups, not semantic
        # judgments. Do not send them to Jev or override a backend match.
        if any(
            query.casefold() in str(patient.get(field) or "").casefold()
            for patient in patients
            for field in ("patient_number", "phone", "email")
            if patient.get(field)
        ):
            return []
        return [patient["name"].strip() for patient in patients]

    if tool_name in {"search_medicines", "search_clinic_knowledge", "search_call_conversations"}:
        key = {
            "search_medicines": "medicines",
            "search_clinic_knowledge": "results",
            "search_call_conversations": "calls",
        }[tool_name]
        items = result.get(key)
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            return []
        if tool_name == "search_medicines":
            return [
                f"{item.get('name') or ''} {item.get('strength') or ''} {item.get('form') or ''} {item.get('category') or ''}".strip()[:256]
                for item in items
            ]
        if tool_name == "search_clinic_knowledge":
            return [f"{item.get('title') or ''} {item.get('question') or ''}".strip()[:256] for item in items]
        # Only a short summary/purpose leaves the service, never transcripts.
        return [
            f"{item.get('participant') or ''} {item.get('requested_purpose') or ''} {item.get('summary') or ''}".strip()[:256]
            for item in items
        ]

    rows = result.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
        return []
    if tool_name == "find_case":
        if any(len(row) <= 3 for row in rows):
            return []
        # The app backend searches identifier fields first. Jev only judges
        # patient-name ambiguity after the identifier search found no match.
        if any(
            query.casefold() in str(row[index]).casefold()
            for row in rows
            for index in (0, 1, 2)
            if len(row) > index and row[index] != "-"
        ):
            return []
        return [str(row[3]).strip() for row in rows]
    if tool_name == "find_doctor":
        if any(len(row) <= 3 for row in rows):
            return []
        if any(
            query.casefold() in str(row[index]).casefold()
            for row in rows
            for index in (3,)
            if len(row) > index and row[index] != "-"
        ):
            return []
        return [f"{row[0]} ({row[1]})"[:256] for row in rows]
    if tool_name == "staff_list":
        if any(len(row) <= 1 for row in rows):
            return []
        return [f"{row[0]} ({row[1]})"[:256] for row in rows]
    return []


async def suggest_search_match(
    tool_name: str, query: Any, result: dict[str, Any]
) -> TextMatchResult | None:
    """Return Jev's best-match suggestion, or None when no decision is needed.

    Callers meter the returned decision and may attach the selected index as
    advisory metadata. They must keep the original candidate list and tenant
    scope intact.
    """
    if not isinstance(query, str) or not query.strip() or len(query) > 200:
        return None
    labels = _candidate_labels(tool_name, query.strip(), result)
    if not 2 <= len(labels) <= 20 or any(not label or len(label) > 256 for label in labels):
        return None
    if len({label.casefold() for label in labels}) != len(labels):
        # Indistinguishable display labels cannot justify picking one record.
        return None
    return await jev_agent.match_text(query.strip(), labels)


def with_search_suggestion(
    result: dict[str, Any], suggestion: TextMatchResult
) -> dict[str, Any]:
    """Add compact advice while preserving original candidates and row IDs."""
    return {
        **result,
        "jev_match": {
            "row_index": suggestion.index,
            "confidence": suggestion.confidence,
            "advisory_only": True,
        },
    }
