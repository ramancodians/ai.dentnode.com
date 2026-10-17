"""Bounded, read-only hourly summary from D10-authorized aggregate facts.

No database, patient text, tool calls or model-selected actions enter this module.
"""
import time
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.jev import JEV_MODEL, jev_agent
from agent.d10.usage_outbox import build_model_usage_event

FEATURE = "clinic_hourly_summary"
FACT_LABELS = {
    "bookings": "bookings", "reschedules": "reschedules", "cancellations": "cancellations",
    "completed": "completed visits", "no_shows": "no-shows", "patient_activity": "patient activity records",
    "calls_recorded": "cached calls", "missed_calls_recorded": "recorded missed calls",
    "work_item_changes": "work updates", "failed_jobs": "failed jobs",
    "open_work_items": "currently open work items", "overdue_work_items": "currently overdue work items",
}


class SummaryCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bookings: int = Field(ge=0, le=1_000_000)
    reschedules: int = Field(ge=0, le=1_000_000)
    cancellations: int = Field(ge=0, le=1_000_000)
    completed: int = Field(ge=0, le=1_000_000)
    no_shows: int = Field(ge=0, le=1_000_000)
    patient_activity: int = Field(ge=0, le=1_000_000)
    calls_recorded: int = Field(ge=0, le=1_000_000)
    missed_calls_recorded: int | None = Field(ge=0, le=1_000_000)
    work_item_changes: int = Field(ge=0, le=1_000_000)
    failed_jobs: int = Field(ge=0, le=1_000_000)
    open_work_items: int = Field(ge=0, le=1_000_000)
    overdue_work_items: int = Field(ge=0, le=1_000_000)


class SummaryUsageContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clinic_id: str = Field(min_length=1, max_length=200)
    user_id: str = Field(min_length=1, max_length=200)
    job_id: str = Field(min_length=1, max_length=200)
    summary_id: str = Field(min_length=1, max_length=200)
    reservation_id: str = Field(min_length=1, max_length=200)


class ClinicSummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    counts: SummaryCounts
    action_types: list[Literal["missed_calls", "overdue_work", "open_work", "failed_jobs"]] = Field(max_length=4)
    usage_context: SummaryUsageContext


class ClinicSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=200)


async def generate_clinic_summary(body: ClinicSummaryRequest, outbox) -> ClinicSummaryResponse:
    if not any(body.counts.model_dump().values()):
        return ClinicSummaryResponse(summary="No new activity recorded in the previous hour.")
    usage = body.usage_context
    claim_key = f"clinic-summary:{usage.clinic_id}:{usage.summary_id}"
    if not await outbox.claim_model_execution(claim_key):
        raise ValueError("Summary generation already claimed; read the D10 cached summary")
    context = SimpleNamespace(
        clinic_id=usage.clinic_id, user_id=usage.user_id,
        actor_id="system:clinic-hourly-summary", conversation_id=None,
        source_message_id=None, correlation_id=f"clinic-summary:{usage.summary_id}",
        causation_id=usage.job_id, reservation_id=usage.reservation_id,
    )
    started = time.monotonic()
    result = None
    try:
        values = body.counts.model_dump()
        candidates = {key: value for key, value in values.items() if value is not None and value > 0}
        result = await jev_agent.decide(
            state={"recorded_counts": candidates, "window": "previous hour; open and overdue counts are current", "action_types": body.action_types},
            questions={key: {"type": "score", "instructions": f"How important is {key} to a short clinic operations summary? Prioritize missed calls, overdue work and failures. Do not choose or perform actions.", "criteria": ["routine", "useful", "important", "urgent"]} for key in candidates},
            timeout_secs=10,
        )
        ranked = sorted(candidates, key=lambda key: -float(result.answers[key]["score"]))
        parts = []
        for key in ranked:
            part = f"{values[key]} {FACT_LABELS[key]}"
            if len("Recorded: " + "; ".join([*parts, part]) + ".") <= 200:
                parts.append(part)
        return ClinicSummaryResponse(summary="Recorded: " + "; ".join(parts) + ".")
    finally:
        # Record spend even when output validation fails. Durable delivery uses one
        # summary identity; HTTP/worker retries must not debit the clinic twice.
        event = build_model_usage_event(
            feature=FEATURE, context=context, model_call_index=1, attempt=1,
            provider=(result.provider or "openrouter") if result else "openrouter",
            model=result.model if result else JEV_MODEL,
            provider_request_id=result.request_id if result else None,
            usage={"prompt_tokens": result.input_tokens, "completion_tokens": result.output_tokens, "total_tokens": result.input_tokens + result.output_tokens} if result else None, raw_usage=None,
            latency_ms=result.latency_ms if result else int((time.monotonic() - started) * 1000),
            status="ok" if result else "error", cost_usd=result.cost_usd if result else None,
            error_code=None if result else "SUMMARY_MODEL_UNAVAILABLE",
        )
        event["job_id"] = usage.job_id
        event["idempotency_key"] = f"clinic-summary:{usage.clinic_id}:{usage.summary_id}"
        await outbox.enqueue(event)
