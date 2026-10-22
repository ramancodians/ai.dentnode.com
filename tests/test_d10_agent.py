"""Endpoint and orchestration tests for the isolated D10 Agent."""

import json
import zoneinfo

import pytest

from agent.d10.context import D10RequestContext
from agent.openrouter import ChatResult
from tests.conftest import D10_TEST_KEY

_HEADERS = {"x-d10-internal-key": D10_TEST_KEY}
_CONTEXT = {
    "clinic_id": "clinic-1",
    "user_id": "user-1",
    "conversation_id": "conversation-1",
    "actor_id": "staff-1",
    "actor_role": "DENTIST",
    "timezone": "America/New_York",
    "source_message_id": "wamid.1",
    "correlation_id": "corr-1",
    "causation_id": "cause-1",
    "reservation_id": "reserve-1",
}
_BODY = {"message": "Remind me tomorrow", "context": _CONTEXT}


def _events(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line]


def test_d10_endpoint_requires_dedicated_key(client):
    assert client.post("/d10/agent/run", json=_BODY).status_code == 401
    assert (
        client.post(
            "/d10/agent/run",
            json=_BODY,
            headers={"x-internal-key": D10_TEST_KEY},
        ).status_code
        == 401
    )


def test_d10_endpoint_passes_trusted_context_and_returns_ndjson(client, monkeypatch):
    import server

    received = {}

    async def _mock(**kwargs):
        received.update(kwargs)
        yield {"type": "delta", "text": "Scheduled."}
        yield {"type": "done", "message": "Scheduled.", "usage": {"model_calls": 1}}

    monkeypatch.setattr(server, "run_d10_turn", _mock)
    response = client.post("/d10/agent/run", json=_BODY, headers=_HEADERS)

    assert response.status_code == 200
    assert "ndjson" in response.headers["content-type"]
    assert response.headers["x-correlation-id"] == "corr-1"
    assert _events(response)[-1]["message"] == "Scheduled."
    assert received["context"].clinic_id == "clinic-1"
    assert received["context"].timezone == "America/New_York"


def test_d10_endpoint_rejects_unknown_timezone(client):
    body = {**_BODY, "context": {**_CONTEXT, "timezone": "India/Definitely-Not-Real"}}
    assert client.post("/d10/agent/run", json=body, headers=_HEADERS).status_code == 422


# Legacy IANA "backward" links — every one of these is a real value a browser
# reports. Chrome on Windows says Asia/Calcutta, not Asia/Kolkata.
_LEGACY_TZ_ALIASES = ["Asia/Calcutta", "US/Eastern", "Europe/Kiev", "Asia/Rangoon"]


@pytest.mark.parametrize("alias", _LEGACY_TZ_ALIASES)
def test_d10_endpoint_accepts_legacy_timezone_aliases(client, alias):
    """Legacy aliases must validate using ONLY the bundled tzdata package.

    python:3.12-slim ships a /usr/share/zoneinfo carrying the canonical zones but
    not the backward links, and `zoneinfo` consults the tzdata package only when
    the name is missing from TZPATH. CI runners have a complete system database,
    so asserting against the default TZPATH would pass with tzdata uninstalled and
    hide the exact bug this guards — emptying TZPATH is what reproduces the image.

    Regression: without tzdata these 422'd, and D10 rendered that to the user as
    "The assistant is unavailable right now."
    """
    zoneinfo.reset_tzpath([])
    try:
        body = {**_BODY, "context": {**_CONTEXT, "timezone": alias}}
        assert client.post("/d10/agent/run", json=body, headers=_HEADERS).status_code != 422
    finally:
        zoneinfo.reset_tzpath()


class _Outbox:
    def __init__(self):
        self.events = []

    async def enqueue(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_turn_records_each_model_call_and_keeps_identity_out_of_arguments(monkeypatch):
    from agent.d10 import runner

    model_results = iter(
        [
            ChatResult(
                text="",
                model="provider/model-v1",
                provider="provider-x",
                request_id="req-1",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cached_input_tokens": 40,
                    "reasoning_tokens": 5,
                },
                raw_usage={"prompt_tokens": 100, "cost": 0.0012},
                cost_usd=0.0012,
                latency_ms=45,
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "schedule_reminder",
                                "arguments": json.dumps(
                                    {"when": "2026-08-23T09:00:00-04:00"}
                                ),
                            },
                        }
                    ],
                },
            ),
            ChatResult(
                text="I’ll remind you tomorrow at 9:00 AM EDT.",
                model="provider/model-v1",
                provider="provider-x",
                request_id="req-2",
                usage={
                    "prompt_tokens": 130,
                    "completion_tokens": 15,
                    "total_tokens": 145,
                },
                raw_usage={"prompt_tokens": 130, "cost": 0.0015},
                cost_usd=0.0015,
                latency_ms=55,
                message={
                    "role": "assistant",
                    "content": "I’ll remind you tomorrow at 9:00 AM EDT.",
                },
            ),
        ]
    )
    tool_invocation = {}

    async def _catalog(_context):
        return [
            {
                "type": "function",
                "function": {
                    "name": "schedule_reminder",
                    "description": "Schedule a reminder",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    async def _model(**_kwargs):
        return next(model_results)

    async def _tool(**kwargs):
        tool_invocation.update(kwargs)
        return {"job_id": "job-1", "status": "scheduled"}

    monkeypatch.setattr(runner, "fetch_tool_catalog", _catalog)
    monkeypatch.setattr(runner, "chat_completion", _model)
    monkeypatch.setattr(runner, "call_tool", _tool)
    outbox = _Outbox()
    context = D10RequestContext(**_CONTEXT)

    events = [
        event
        async for event in runner.run_d10_turn(
            message="Remind me tomorrow at 9",
            context=context,
            history=None,
            outbox=outbox,
        )
    ]

    assert len(outbox.events) == 2
    assert [event["provider_request_id"] for event in outbox.events] == ["req-1", "req-2"]
    assert outbox.events[0]["cached_input_tokens"] == 40
    assert outbox.events[0]["reasoning_tokens"] == 5
    assert outbox.events[0]["provider_cost_micros_usd"] == 1200
    assert tool_invocation["parameters"] == {"when": "2026-08-23T09:00:00-04:00"}
    assert tool_invocation["context"].clinic_id == "clinic-1"
    done = events[-1]
    assert done["type"] == "done"
    assert done["usage"] == {
        "model_calls": 2,
        "input_tokens": 230,
        "output_tokens": 35,
        "total_tokens": 265,
        "cached_input_tokens": 40,
        "reasoning_tokens": 5,
    }


@pytest.mark.asyncio
async def test_failed_model_call_is_still_metered(monkeypatch):
    from agent.d10 import runner
    from agent.openrouter import OpenRouterError

    async def _catalog(_context):
        return []

    async def _model(**_kwargs):
        raise OpenRouterError("secret provider detail")

    monkeypatch.setattr(runner, "fetch_tool_catalog", _catalog)
    monkeypatch.setattr(runner, "chat_completion", _model)
    outbox = _Outbox()
    events = [
        event
        async for event in runner.run_d10_turn(
            message="hello",
            context=D10RequestContext(**_CONTEXT),
            history=None,
            outbox=outbox,
        )
    ]

    assert outbox.events[0]["status"] == "error"
    assert outbox.events[0]["error_code"] == "OpenRouterError"
    assert "secret provider detail" not in json.dumps(events)


def test_model_cannot_override_trusted_tool_context():
    from agent.d10.client import D10ToolError
    from agent.d10.runner import _tool_arguments

    with pytest.raises(D10ToolError, match="trusted context"):
        _tool_arguments(
            {
                "function": {
                    "arguments": json.dumps(
                        {"clinicId": "other-clinic", "when": "tomorrow"}
                    )
                }
            }
        )


def test_system_prompt_relays_data_errors_instead_of_calling_the_feature_broken():
    """A tool error about the clinic's own data must reach the user as such.

    Regression guard for a real incident: reminder calls failed because every
    patient's phone was stored without a country code, and the gateway returned
    a clear message saying so. The prompt's blanket "never expose raw errors"
    made the model report it as "the automated reminder call feature is
    currently unavailable", which sent the operator to check deploys and logs
    instead of fixing one field. The prompt must still withhold internals.
    """
    from agent.d10.runner import SYSTEM_INSTRUCTION

    prompt = SYSTEM_INSTRUCTION.lower()
    # Still refuses to leak the things that are genuinely sensitive.
    for secret in ("credentials", "internal ids", "prompts", "tool internals"):
        assert secret in prompt
    # But no longer suppresses actionable validation messages wholesale.
    assert "raw errors" not in prompt
    assert "relay that reason in plain language" in prompt
    assert "never describe a working feature as unavailable" in prompt
