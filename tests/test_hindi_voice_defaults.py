from agent.calling_prompt import calling_system_prompt
from voice_call.config import CallConfig, opening_instructions
from voice_call.metering import cost_lines
from types import SimpleNamespace

import pytest


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
