from types import SimpleNamespace

import pytest

from voice_call.accounting import VoiceAccounting
from voice_call.config import CallConfig
from voice_call.metering import cost_lines, d10_events


@pytest.mark.asyncio
async def test_repeated_generation_metrics_never_duplicate_the_provider_charge(monkeypatch):
    async def prices(): return {"anthropic/claude-haiku-4.5": (0.000001, 0.000005)}
    monkeypatch.setattr("voice_call.metering.openrouter_token_prices", prices)
    usage = SimpleNamespace(model_usage=[SimpleNamespace(type="llm_usage", input_tokens=20, output_tokens=4)])
    metric = {"type": "llm_metrics", "request_id": "gen-one", "prompt_tokens": 10, "completion_tokens": 2}
    lines = await cost_lines(CallConfig(), usage, session_seconds=1, sip_seconds=0,
        metrics=[metric, metric], receipts={"gen-one": {"id": "gen-one", "total_cost": 0.012}})
    llm = next(line for line in lines if line.kind == "llm")
    assert llm.cost_usd == 0.012
    assert llm.cost_source == "provider_reported"


@pytest.mark.asyncio
async def test_missing_usage_is_unknown_and_self_hosted_compute_is_unallocated(monkeypatch):
    async def prices(): return {}
    monkeypatch.setattr("voice_call.metering.openrouter_token_prices", prices)
    lines = await cost_lines(CallConfig(), SimpleNamespace(model_usage=[]), session_seconds=45, sip_seconds=0)
    assert {line.kind for line in lines} == {"llm", "stt", "tts", "agent", "telephony"}
    assert all(line.cost_usd is None for line in lines)
    agent = next(line for line in lines if line.kind == "agent")
    assert agent.cost_source == "unallocated"
    context = {"clinic_id": "clinic", "user_id": "user", "actor_id": "actor", "conversation_id": "conv", "source_message_id": "msg", "correlation_id": "call"}
    events = d10_events(context, lines)
    billed = next(event for event in events if event["event_type"] == "ai.voice_agent_session")
    assert billed["pricing"]["charged_credits_micros"] > 0
    assert billed["provider_cost_micros_usd"] is None
    failed_events = d10_events(context, lines, charge_session=False)
    assert all(event["pricing"]["charged_credits_micros"] == 0 for event in failed_events)


@pytest.mark.asyncio
async def test_checkpoint_storage_failure_does_not_abort_the_call():
    cfg = CallConfig(usage_context={"clinic_id": "clinic", "user_id": "user", "actor_id": "actor", "conversation_id": "conv", "source_message_id": "msg", "correlation_id": "call"})
    accounting = VoiceAccounting(cfg, room="room", job_id="job", dispatch_id="dispatch")
    async def unavailable(): raise OSError("storage unavailable")
    accounting.outbox = SimpleNamespace(initialize=unavailable)
    await accounting.checkpoint("worker_assigned")
    assert accounting.failed_checkpoints == 1


@pytest.mark.asyncio
async def test_pinned_sdk_metrics_hook_records_ids_without_speech_text():
    from livekit.agents.metrics import LLMMetrics
    from livekit.plugins.openai import LLM
    llm = LLM(model="test-model", api_key="test", base_url="https://example.invalid")
    accounting = VoiceAccounting(CallConfig(), room="room", job_id="job", dispatch_id="dispatch")
    llm.on("metrics_collected", accounting.capture)
    llm.emit("metrics_collected", LLMMetrics(label="test", request_id="gen-test", timestamp=1,
        duration=1, ttft=0.2, cancelled=False, completion_tokens=2, prompt_tokens=10,
        prompt_cached_tokens=3, reasoning_tokens=1, total_tokens=12, tokens_per_second=2))
    await accounting.drain()
    assert accounting.metrics[0]["request_id"] == "gen-test"
    assert accounting.metrics[0]["prompt_cached_tokens"] == 3
    assert "text" not in accounting.metrics[0]
    await llm.aclose()
