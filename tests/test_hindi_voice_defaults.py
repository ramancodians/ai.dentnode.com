from agent.calling_prompt import calling_system_prompt
from voice_call.config import CallConfig, opening_instructions
from voice_call.metering import cost_lines
from types import SimpleNamespace

import pytest
from voice_call.callbacks import CallCallback, notify


def test_default_phone_reply_and_livekit_opening_use_hindi():
    assert "Hindi in Devanagari script" in calling_system_prompt("Confirm the callback")
    cfg = CallConfig()
    assert cfg.language == "hi"
    assert "Speak in Hindi" in opening_instructions(cfg)


def test_explicit_english_voice_choice_is_preserved():
    cfg = CallConfig.from_metadata('{"language":"en"}')
    assert cfg.language == "en"
    assert "Speak in Hindi" not in opening_instructions(cfg)
    assert "in English" in calling_system_prompt("Confirm the callback", language=cfg.language)


def test_dispatch_can_shorten_but_never_extend_voice_duration_cap():
    assert CallConfig.from_metadata('{}').max_call_seconds == 300
    assert CallConfig.from_metadata('{"max_call_seconds":60}').max_call_seconds == 60
    for value in ('0', '301', '-1', 'true', '60.5', '"60"'):
        with pytest.raises(ValueError, match="max_call_seconds"):
            CallConfig.from_metadata('{"max_call_seconds":' + value + '}')


@pytest.mark.asyncio
async def test_final_callback_carries_frozen_ai_duration_without_changing_start_claim(monkeypatch):
    bodies = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, **kwargs):
            bodies.append(kwargs["json"])
            return SimpleNamespace(status_code=200, is_success=True)

    monkeypatch.setattr("voice_call.callbacks.httpx.AsyncClient", Client)
    callback = CallCallback("https://calling.example/internal/ai-answering/call-1", "private-test-token")
    assert await notify(callback, "started")
    assert await notify(callback, "completed", ai_duration_seconds=59.9)
    assert bodies == [{"status": "started"}, {"status": "completed", "aiDurationSeconds": 59.9}]
    with pytest.raises(ValueError, match="final AI duration"):
        await notify(callback, "completed", ai_duration_seconds=float("inf"))


@pytest.mark.asyncio
async def test_elevenlabs_hindi_voice_is_metered_by_characters(monkeypatch):
    async def no_llm_prices():
        return {}

    monkeypatch.setattr("voice_call.metering.openrouter_token_prices", no_llm_prices)
    cfg = CallConfig(tts_model="elevenlabs/eleven_flash_v2_5", tts_voice="voice-id")
    usage = SimpleNamespace(model_usage=[SimpleNamespace(type="tts_usage", characters_count=1000, audio_duration=30)])
    lines = await cost_lines(cfg, usage, session_seconds=0, sip_seconds=0)
    tts = next(line for line in lines if line.kind == "tts")
    assert (tts.provider, tts.quantity, tts.cost_usd) == ("elevenlabs", 1000, 0.04)
