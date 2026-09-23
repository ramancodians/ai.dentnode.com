"""Provider-native usage fields retained by the OpenRouter primitive."""

import io
import logging
from types import SimpleNamespace

import httpx
import pytest
import respx
from httpx import Response

from agent.config import settings
from agent.openrouter import OpenRouterError, chat_completion
from tests.conftest import TEST_KEY


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


@pytest.mark.asyncio
@respx.mock
async def test_chat_completion_provider_error_is_sanitized(monkeypatch):
    secret = "SECRET_PROVIDER_RESPONSE_MARKER"
    monkeypatch.setattr(settings, "openrouter_api_base", "https://openrouter.test/api/v1")
    respx.post("https://openrouter.test/api/v1/chat/completions").mock(
        return_value=Response(429, text=f"provider body contains {secret}")
    )

    with pytest.raises(OpenRouterError) as exc_info:
        await chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            model="openrouter/google/gemini-test",
        )

    assert str(exc_info.value) == "OpenRouter request failed"
    assert exc_info.value.code == "provider_http_error"
    assert exc_info.value.status_code == 429
    assert secret not in str(exc_info.value)


def test_chat_completion_provider_body_never_reaches_endpoint_logs(
    client, monkeypatch
):
    import server
    from agent.openrouter import chat_completion as real_chat_completion

    secret = "SECRET_PROVIDER_RESPONSE_MARKER"
    provider_response = SimpleNamespace(
        status_code=503,
        text=f"provider body contains {secret}",
    )

    async def fake_post(*_args, **_kwargs):
        return provider_response

    async def generate_with_real_chat(**_kwargs):
        return await real_chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            model="openrouter/google/gemini-test",
        )

    monkeypatch.setattr(server, "generate_insights", generate_with_real_chat)
    monkeypatch.setattr("agent.openrouter.httpx.AsyncClient.post", fake_post)
    monkeypatch.setattr(server, "report_usage", lambda **_kwargs: None)
    monkeypatch.setattr(server, "_fire_and_forget", lambda _coro: None)

    records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = CaptureHandler()
    server.logger.addHandler(handler)
    try:
        response = client.post(
            "/insights",
            json={"lab_id": "lab-123", "data": {}, "question": "summary"},
            headers={"x-internal-key": TEST_KEY},
        )
    finally:
        server.logger.removeHandler(handler)

    assert response.status_code == 502
    rendered_logs = " ".join(
        f"{record.getMessage()} {record.__dict__!r}" for record in records
    )
    assert secret not in rendered_logs


@pytest.mark.asyncio
async def test_chat_completion_transport_error_traceback_is_sanitized(monkeypatch):
    secret = "SECRET_TRANSPORT_URL_MARKER"

    async def fail_post(*_args, **_kwargs):
        raise httpx.ConnectError(
            f"failed to connect to https://user:{secret}@provider.invalid"
        )

    monkeypatch.setattr("agent.openrouter.httpx.AsyncClient.post", fail_post)

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    logger = logging.getLogger("tests.openrouter.transport")
    original_level = logger.level
    original_propagate = logger.propagate
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        with pytest.raises(OpenRouterError) as exc_info:
            try:
                await chat_completion(
                    messages=[{"role": "user", "content": "hello"}],
                    model="openrouter/google/gemini-test",
                )
            except OpenRouterError:
                logger.exception("OpenRouter transport request failed")
                raise
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate

    assert exc_info.value.code == "transport_error"
    assert secret not in log_stream.getvalue()
