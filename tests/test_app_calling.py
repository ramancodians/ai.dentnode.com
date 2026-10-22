from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.openrouter import ChatResult, OpenRouterError


@pytest.fixture
def voice(client, monkeypatch):
    import server
    monkeypatch.setattr(server.settings, "calling_voice_internal_key", "test-calling-key")
    model = AsyncMock(return_value=ChatResult(text="The lab team will follow up.", model="test-model", cost_usd=0.001))
    app_usage = AsyncMock()
    d10_usage = AsyncMock()
    monkeypatch.setattr(server, "chat_completion", model)
    monkeypatch.setattr(server, "report_usage", app_usage)
    monkeypatch.setattr(server.d10_usage_outbox, "enqueue", d10_usage)
    return SimpleNamespace(client=client, model=model, app_usage=app_usage, d10_usage=d10_usage)


def request(platform="app"):
    context = {"platform": platform, "user_id": "user-1", "actor_id": "user-1",
        "conversation_id": "draft-1", "source_message_id": "draft-1", "correlation_id": "call-1", "model_call_index": 1}
    context["lab_id" if platform == "app" else "clinic_id"] = "tenant-1"
    return {"objective": "Ask about crown approval", "recipient_name": "Dr Example", "speech": "Please ask the team.", "usage_context": context}


def post(voice, body, key="test-calling-key"):
    return voice.client.post("/calling/voice/reply", json=body, headers={"x-calling-internal-key": key})


def test_app_reply_never_uses_d10_wallet(voice):
    response = post(voice, request())
    assert response.status_code == 200
    voice.d10_usage.assert_not_called()
    values = voice.app_usage.call_args.kwargs
    assert values["lab_id"] == "tenant-1"
    assert values["feature"] == "app_ai_call_reply"
    assert values["meta"] == {"call_id": "call-1", "included_in_app_call": True}
    assert "dental lab" in voice.model.call_args.kwargs["messages"][0]["content"]


def test_legacy_d10_context_still_uses_d10_usage_outbox(voice):
    body = request("d10")
    del body["usage_context"]["platform"]
    assert post(voice, body).status_code == 200
    voice.app_usage.assert_not_called()
    voice.d10_usage.assert_awaited_once()


@pytest.mark.parametrize("platform", ["app", "d10"])
def test_conflicting_platform_scope_rejected_before_model_or_usage(voice, platform):
    body = request(platform)
    body["usage_context"]["clinic_id" if platform == "app" else "lab_id"] = "other-tenant"
    assert post(voice, body).status_code == 422
    voice.model.assert_not_called()
    voice.app_usage.assert_not_called()
    voice.d10_usage.assert_not_called()


def test_app_provider_failure_is_metered_only_to_app(voice):
    voice.model.side_effect = OpenRouterError()
    assert post(voice, request()).status_code == 502
    assert voice.app_usage.call_args.kwargs["status"] == "error"
    voice.d10_usage.assert_not_called()


def test_unauthenticated_caller_cannot_choose_app_tenant(voice):
    assert post(voice, request(), "wrong-key").status_code == 401
    voice.model.assert_not_called()


def test_call_tool_never_exposes_confirmation_or_destination():
    import inspect
    from agent.tools import LABY_TOOLS, call_client
    assert call_client in LABY_TOOLS
    assert set(inspect.signature(call_client).parameters) == {"tool_context", "doctor_name", "objective", "doctor_clinic"}
    assert {"wallet_balance", "call_history", "call_summary"}.issubset({tool.__name__ for tool in LABY_TOOLS})
