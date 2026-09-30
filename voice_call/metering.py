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
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from agent.config import settings
from agent.d10.usage_outbox import UsageOutbox, build_model_usage_event, build_unit_usage_event
from agent.usage import report_usage

from .config import CallConfig
from .costs import (
    AGENT_SESSION_USD_PER_MIN, PRICE_TABLE_VERSION, SIP_USD_PER_MIN, STT_USD_PER_MIN,
    TTS_USD_PER_MIN, llm_cost, openrouter_token_prices, per_minute,
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


async def cost_lines(cfg: CallConfig, usage: Any, *, session_seconds: float,
                     sip_seconds: float) -> List[CostLine]:
    # Models come from the call config, not from usage records, whose naming
    # differs per plugin; the config is exactly what the call was billed for.
    prices = await openrouter_token_prices()
    llm_in = llm_out = 0
    stt_seconds = tts_seconds = 0.0
    tts_chars = 0
    for u in getattr(usage, "model_usage", []):
        kind = getattr(u, "type", None)
        if kind == "llm_usage":
            llm_in += u.input_tokens
            llm_out += u.output_tokens
        elif kind == "stt_usage":
            stt_seconds += u.audio_duration
        elif kind == "tts_usage":
            tts_chars += u.characters_count
            tts_seconds += u.audio_duration
    lines = [
        CostLine("llm", "openrouter", cfg.reply_model, llm_in + llm_out, "token",
                 llm_cost(prices, cfg.reply_model, llm_in, llm_out), llm_in, llm_out),
        CostLine("stt", "livekit", cfg.stt_model, int(stt_seconds * 1000), "millisecond",
                 per_minute(STT_USD_PER_MIN, cfg.stt_model, stt_seconds)),
        CostLine("tts", "livekit", cfg.tts_model, tts_chars, "character",
                 per_minute(TTS_USD_PER_MIN, cfg.tts_model, tts_seconds)),
        CostLine("agent", "livekit", "agent-session", int(session_seconds * 1000), "millisecond",
                 AGENT_SESSION_USD_PER_MIN * session_seconds / 60.0),
    ]
    if sip_seconds > 0:
        lines.append(CostLine("telephony", "livekit_sip", "sip", int(sip_seconds * 1000),
                              "millisecond", SIP_USD_PER_MIN * sip_seconds / 60.0))
    return [line for line in lines if line.quantity > 0]


_UNIT_EVENT = {"stt": "ai.speech_to_text", "tts": "ai.text_to_speech",
               "agent": "ai.voice_agent_session", "telephony": "telephony.call"}


def d10_events(ctx: Dict[str, str], lines: List[CostLine]) -> List[Dict[str, Any]]:
    context = SimpleNamespace(**ctx, causation_id=ctx["source_message_id"], reservation_id=None)
    raw = {"cost_source": "price_table", "price_table_version": PRICE_TABLE_VERSION}
    events = []
    for sequence, line in enumerate(lines, start=1):
        if line.kind == "llm":
            events.append(build_model_usage_event(
                feature=FEATURE, context=context, model_call_index=sequence, attempt=1,
                provider="openrouter", model=line.model, provider_request_id=None,
                usage={"prompt_tokens": line.input_tokens, "completion_tokens": line.output_tokens,
                       "total_tokens": line.quantity},
                raw_usage=raw, latency_ms=0, status="ok", cost_usd=line.cost_usd,
            ))
        else:
            events.append(build_unit_usage_event(
                context=context, event_type=_UNIT_EVENT[line.kind], quantity=line.quantity,
                feature=FEATURE, provider=line.provider, model=line.model, sequence=sequence,
                cost_usd=line.cost_usd, cost_source="price_table",
                raw_usage={"price_table_version": PRICE_TABLE_VERSION},
            ))
    return events


async def book_call(cfg: CallConfig, usage: Any, *, room: str, session_seconds: float,
                    sip_seconds: float) -> None:
    lines = await cost_lines(cfg, usage, session_seconds=session_seconds, sip_seconds=sip_seconds)
    if cfg.usage_context:
        outbox = UsageOutbox(voice_outbox_path())
        await outbox.initialize()
        for event in d10_events(cfg.usage_context, lines):
            await outbox.enqueue(event)
        await outbox.flush_once()  # best effort; the server's loop retries leftovers
        return
    for line in lines:
        await report_usage(
            feature=FEATURE, lab_id=_PLATFORM_LAB_ID, model=line.model,
            usage=({"prompt_tokens": line.input_tokens, "completion_tokens": line.output_tokens,
                    "total_tokens": line.quantity} if line.kind == "llm" else None),
            cost=line.cost_usd, cost_source="estimated",
            meta={"room": room, "source": cfg.source, "component": line.kind,
                  "quantity": line.quantity, "unit": line.unit,
                  "price_table_version": PRICE_TABLE_VERSION},
        )
