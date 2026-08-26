"""Provider-native usage fields retained by the OpenRouter primitive."""

import pytest
import respx
from httpx import Response

from agent.config import settings
from agent.openrouter import chat_completion


@pytest.mark.asyncio
@respx.mock
async def test_chat_completion_preserves_request_provider_and_detailed_usage(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_base", "https://openrouter.test/api/v1")
    route = respx.post("https://openrouter.test/api/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "generation-123",
                "provider": "Google",
                "model": "google/gemini-test",
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "total_tokens": 125,
                    "cost": 0.0025,
                    "prompt_tokens_details": {
                        "cached_tokens": 40,
                        "image_tokens": 8,
                        "audio_tokens": 2,
                    },
                    "completion_tokens_details": {"reasoning_tokens": 6},
                },
            },
        )
    )

    result = await chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        model="openrouter/google/gemini-test",
    )

    assert route.called
    assert result.request_id == "generation-123"
    assert result.provider == "Google"
    assert result.model == "google/gemini-test"
    assert result.usage["cached_input_tokens"] == 40
    assert result.usage["reasoning_tokens"] == 6
    assert result.usage["image_input_units"] == 8
    assert result.usage["audio_input_units"] == 2
    assert result.raw_usage["cost"] == 0.0025
