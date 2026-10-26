"""Book one finished voice call's spend, component by component.

A call dispatched with a clinic ``usage_context`` (Calling Service will pass
one) goes to that clinic's D10 usage ledger; anything else — Voice Lab — is
platform spend in the Node ledger. Each component is its own event so the
owner report can show where a call's money went:

  reply LLM (tokens) · speech-to-text (audio ms) · TTS (characters) ·
  agent session (ms) · phone line (ms, only when a SIP caller was present)

D10 events go through a durable SQLite outbox. The worker flushes it when the
call ends; the API server also runs a flush loop over the same file, so an
event that missed D10 is retried even if no further call comes. Deliveries are
idempotent in D10, so a double flush is harmless.
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from agent.config import settings
from agent.d10.usage_outbox import UsageOutbox, build_model_usage_event, build_unit_usage_event
from agent.d10.pricing import CONFIG as RETAIL_CONFIG, VERSION as RETAIL_VERSION
from agent.usage import report_usage

from .config import CallConfig, stt_language
from .costs import (
    PRICE_TABLE_VERSION, SIP_USD_PER_MIN, STT_USD_PER_MIN,
    ELEVENLABS_FLASH_USD_PER_1K_CHARS, TTS_USD_PER_1K_CHARS, PRICE_SOURCES, llm_cost, openrouter_token_prices, openrouter_price_snapshot, per_minute,
)

logger = logging.getLogger(__name__)

FEATURE = "voice_call"
_PLATFORM_LAB_ID = "__platform__"


def voice_outbox_path() -> str:
    return settings.d10_usage_outbox_path + ".voice"


@dataclass
class CostLine:
    kind: str  # llm | stt | tts | agent | telephony
    provider: str
    model: str
    quantity: int
    unit: str
    cost_usd: Optional[float]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_source: str = "unknown"
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


async def cost_lines(cfg: CallConfig, usage: Any, *, session_seconds: float,
                     sip_seconds: float, metrics: Optional[List[dict]] = None,
                     receipts: Optional[Dict[str, dict]] = None) -> List[CostLine]:
    # Keep requested models and observed plugin names separately. List-price
    # estimation uses the configured route; a receipt proves the actual charge.
    prices = await openrouter_token_prices()
    llm_in = llm_out = 0
    stt_seconds = tts_seconds = 0.0
    tts_chars = 0
    observed = set()
    for u in getattr(usage, "model_usage", []):
        kind = getattr(u, "type", None)
        observed.add(kind)
        if kind == "llm_usage":
            llm_in += u.input_tokens
            llm_out += u.output_tokens
        elif kind == "stt_usage":
            stt_seconds += u.audio_duration
        elif kind == "tts_usage":
            tts_chars += u.characters_count
            tts_seconds += u.audio_duration
    requests = metrics or []
    llm_requests = [m for m in requests if m.get("type") == "llm_metrics"]
    receipts = receipts or {}
    receipt_ids = set(m.get("request_id") for m in llm_requests if m.get("request_id"))
    receipt_complete = bool(llm_requests) and all(m.get("request_id") in receipts for m in llm_requests)
    receipt_complete = receipt_complete and sum(m.get("prompt_tokens", 0) for m in llm_requests) == llm_in and sum(m.get("completion_tokens", 0) for m in llm_requests) == llm_out
    llm_amount = (sum(receipts[request_id]["total_cost"] for request_id in receipt_ids)
                  if receipt_complete else llm_cost(prices, cfg.reply_model, llm_in, llm_out) if "llm_usage" in observed else None)
    stt_key = f"{cfg.stt_model}:multi" if stt_language(cfg.stt_model, cfg.language) == "multi" else cfg.stt_model
    stt_amount = per_minute(STT_USD_PER_MIN, stt_key, stt_seconds) if stt_seconds > 0 else None
    tts_rate = ELEVENLABS_FLASH_USD_PER_1K_CHARS if cfg.tts_model.startswith("elevenlabs/") else TTS_USD_PER_1K_CHARS.get(cfg.tts_model)
    tts_amount = tts_rate * tts_chars / 1000 if tts_rate is not None and "tts_usage" in observed else None
    lines = [
        CostLine("llm", "openrouter", cfg.reply_model, llm_in + llm_out, "token",
                 llm_amount, llm_in, llm_out,
                 "provider_reported" if receipt_complete else "price_table" if llm_amount is not None else "unknown",
                 None if receipt_complete else "Generation receipts unavailable or incomplete; list estimate excludes cache/route differences." if llm_amount is not None else "No complete LLM usage/receipt was observed.",
                 {"requests": llm_requests, "receipts": list(receipts.values()), "receipt_complete": receipt_complete,
                  "catalog_snapshot": openrouter_price_snapshot(),
                  "cached_input_tokens": sum(m.get("prompt_cached_tokens", 0) for m in llm_requests),
                  "reasoning_tokens": sum(m.get("reasoning_tokens", 0) for m in llm_requests)}),
        CostLine("stt", "livekit", cfg.stt_model, int(stt_seconds * 1000), "millisecond",
                 stt_amount, cost_source="price_table" if stt_amount is not None else "unknown",
                 reason="Audio duration is observed usage, not the provider's billed connection time. Allowances, rounding and plan unverified.",
                 details={"requests": [m for m in requests if m.get("type") == "stt_metrics"], "rate_usd_per_minute": STT_USD_PER_MIN.get(stt_key), "stt_language": stt_language(cfg.stt_model, cfg.language), "billable_connection_seconds": None}),
        CostLine("tts", "elevenlabs" if cfg.tts_model.startswith("elevenlabs/") else "livekit",
                 cfg.tts_model, tts_chars, "character", tts_amount,
                 cost_source="price_table" if tts_amount is not None else "unknown",
                 reason="Public character rate; supplier plan, quota, voice surcharges and invoice are not reconciled.",
                 details={"requests": [m for m in requests if m.get("type") == "tts_metrics"], "rate_usd_per_1000_characters": tts_rate, "audio_seconds": tts_seconds}),
        # The worker is hosted on DentNode's VPS, not LiveKit Cloud. Duration
        # still drives the unchanged clinic minute price, but is not a cloud
        # hosting invoice. Its supplier compute expense needs fixed allocation.
        CostLine("agent", "self_hosted", "worker-session", int(session_seconds * 1000), "millisecond", None,
                 cost_source="unallocated", reason="Self-hosted VPS compute; LiveKit Cloud agent deployment fee does not apply."),
    ]
    lines.append(CostLine("telephony", "livekit_sip", "sip", int(sip_seconds * 1000),
        "millisecond", SIP_USD_PER_MIN * sip_seconds / 60.0 if sip_seconds > 0 else None,
        cost_source="price_table" if sip_seconds > 0 else "unknown",
        reason="Observed SIP duration; supplier rounding, included minutes and account plan are unverified." if sip_seconds > 0 else "No SIP usage was observed; missing evidence is not proof of zero carrier/transport expense.",
        details={"rate_usd_per_minute": SIP_USD_PER_MIN, "billable_seconds": None}))
    return lines  # Keep unknown and observed-zero components visible.


_UNIT_EVENT = {"stt": "ai.speech_to_text", "tts": "ai.text_to_speech",
               "agent": "ai.voice_agent_session", "telephony": "telephony.call"}


def d10_events(ctx: Dict[str, str], lines: List[CostLine], *, accounting: Optional[dict] = None,
               charge_session: bool = True) -> List[Dict[str, Any]]:
    context = SimpleNamespace(**ctx, causation_id=ctx["source_message_id"], reservation_id=None)
    events = []
    for sequence, line in enumerate(lines, start=1):
        raw = {"cost_source": line.cost_source, "cost_status": line.cost_source.upper(),
               "unknown_reason": line.reason, "price_table_version": PRICE_TABLE_VERSION,
               "price_sources": PRICE_SOURCES, "rates_retrieved_at": "2026-10-25",
               "account_plan": "UNVERIFIED", "allowance_allocation": "UNALLOCATED",
               "supplier_currency": None if line.cost_source == "unallocated" else "USD", "expense_owner": "DENTNODE", "component": line.kind,
               "fx": {"inr_per_usd": RETAIL_CONFIG["inr_per_usd"], "source": "internal_management_convention", "pricing_version": RETAIL_VERSION, "settlement_fx": None},
               "retail_pricing_separate": True, "unit_observation": "OBSERVED" if line.quantity > 0 else "NO_POSITIVE_USAGE_OBSERVED",
               "quantity": line.quantity, "unit": line.unit,
               **(accounting or {}), **line.details}
        raw["requested_model"] = line.model
        raw["observed_models"] = sorted({m["model"] for m in line.details.get("requests", []) if isinstance(m.get("model"), str)})
        if line.kind == "llm":
            events.append(build_model_usage_event(
                feature=FEATURE, context=context, model_call_index=sequence, attempt=1,
                provider="openrouter", model=line.model, provider_request_id=(line.details.get("requests") or [{}])[0].get("request_id") if len(line.details.get("requests", [])) == 1 else None,
                usage={"prompt_tokens": line.input_tokens, "completion_tokens": line.output_tokens,
                       "total_tokens": line.quantity, "cached_input_tokens": line.details.get("cached_input_tokens", 0),
                       "reasoning_tokens": line.details.get("reasoning_tokens", 0)},
                raw_usage=raw, latency_ms=0, status="ok", cost_usd=line.cost_usd,
            ))
        else:
            events.append(build_unit_usage_event(
                context=context, event_type=_UNIT_EVENT[line.kind], quantity=line.quantity,
                feature=FEATURE, provider=line.provider, model=line.model, sequence=sequence,
                cost_usd=line.cost_usd, cost_source=line.cost_source, raw_usage=raw,
            ))
            if line.kind == "agent" and not charge_session:
                # Startup/claim failures retain supplier observations without
                # introducing a new customer minute debit.
                from agent.d10.pricing import price_event
                events[-1]["pricing"] = price_event(event_type="ai.model_call", feature=FEATURE, cost_micros_usd=None)
    return events


async def book_call(cfg: CallConfig, usage: Any, *, room: str, session_seconds: float,
                    sip_seconds: float, metrics: Optional[List[dict]] = None,
                    accounting: Optional[dict] = None, charge_session: bool = True) -> None:
    from .accounting import generation_receipts
    try:
        receipts = await generation_receipts(metrics or [])
    except Exception:
        receipts = {}
        logger.warning("OpenRouter receipts unavailable; preserving original voice usage")
    lines = await cost_lines(cfg, usage, session_seconds=session_seconds, sip_seconds=sip_seconds,
                             metrics=metrics, receipts=receipts)
    accounting = {"room": room, "recorded_at": datetime.now(timezone.utc).isoformat(),
                  "unallocated_components": ["number_rental", "vps_compute", "payment_fees", "subscription_minimums", "tax", "fx_settlement"],
                  "unknown_components": ["recording", "webrtc", "data_transfer", "optional_turn_detection"], **(accounting or {})}
    if cfg.usage_context:
        outbox = UsageOutbox(voice_outbox_path())
        await outbox.initialize()
        for event in d10_events(cfg.usage_context, lines, accounting=accounting, charge_session=charge_session):
            await outbox.enqueue(event)
        try:
            await asyncio.wait_for(outbox.flush_once(), timeout=2)
        except asyncio.TimeoutError:
            # Already persisted. The API process runs a durable .voice flush
            # loop independently of this worker's bounded shutdown window.
            logger.warning("Voice usage delivery deferred to durable outbox loop")
        return
    for line in lines:
        await report_usage(
            feature=FEATURE, lab_id=_PLATFORM_LAB_ID, model=line.model,
            usage=({"prompt_tokens": line.input_tokens, "completion_tokens": line.output_tokens,
                    "total_tokens": line.quantity} if line.kind == "llm" else None),
            # The legacy platform endpoint requires a numeric cost. A zero
            # placeholder is explicitly UNKNOWN/UNALLOCATED, never actual zero.
            cost=line.cost_usd if line.cost_usd is not None else 0, cost_source=line.cost_source,
            meta={"room": room, "source": cfg.source, "component": line.kind,
                  "quantity": line.quantity, "unit": line.unit,
                  "price_table_version": PRICE_TABLE_VERSION, "cost_status": line.cost_source.upper(),
                  "cost_unknown": line.cost_usd is None, "unknown_reason": line.reason,
                  **accounting, **line.details},
        )
