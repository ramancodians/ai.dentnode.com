"""Durability and retry tests for the D10 usage outbox."""

import json
import sqlite3

import httpx
import pytest

from agent.d10.context import D10RequestContext
from agent.d10.usage_outbox import UsageOutbox, build_model_usage_event


def _context() -> D10RequestContext:
    return D10RequestContext(
        clinic_id="clinic-1",
        user_id="user-1",
        conversation_id="conv-1",
        actor_id="actor-1",
        actor_role="DENTIST",
        timezone="UTC",
        source_message_id="wamid.1",
        correlation_id="corr-1",
    )


def _event():
    return build_model_usage_event(
        context=_context(),
        model_call_index=1,
        attempt=1,
        provider="openrouter",
        model="model-v1",
        provider_request_id="provider-req-1",
        usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        raw_usage={"prompt_tokens": 7},
        latency_ms=12,
        status="ok",
        cost_usd=0.000001,
    )


@pytest.mark.asyncio
async def test_enqueue_is_committed_and_idempotent(tmp_path):
    path = tmp_path / "usage.sqlite3"
    outbox = UsageOutbox(str(path))
    await outbox.initialize()
    event = _event()
    await outbox.enqueue(event)
    await outbox.enqueue({**event, "event_id": "different-event-id"})

    assert await outbox.pending_count() == 1
    with sqlite3.connect(path) as connection:
        payload = json.loads(
            connection.execute("SELECT payload FROM d10_usage_outbox").fetchone()[0]
        )
    assert payload["provider_request_id"] == "provider-req-1"
    assert payload["provider_cost_usd"] == "0.000001"


@pytest.mark.asyncio
async def test_flush_delivers_batch_and_marks_rows(tmp_path, monkeypatch):
    path = tmp_path / "usage.sqlite3"
    outbox = UsageOutbox(str(path), endpoint="http://d10.test/usage", internal_key="key")
    await outbox.initialize()
    await outbox.enqueue(_event())
    captured = {}

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured["key"] = request.headers["x-d10-internal-key"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(202, json={"accepted": 1})

    transport = httpx.MockTransport(_handler)
    original_async_client = httpx.AsyncClient

    class _Client:
        def __init__(self, **_kwargs):
            self.client = original_async_client(transport=transport)

        async def __aenter__(self):
            return self.client

        async def __aexit__(self, *_args):
            await self.client.aclose()

    monkeypatch.setattr("agent.d10.usage_outbox.httpx.AsyncClient", _Client)
    assert await outbox.flush_once() == 1
    assert await outbox.pending_count() == 0
    assert captured["key"] == "key"
    assert captured["body"]["events"][0]["event_type"] == "ai.model_call"


@pytest.mark.asyncio
async def test_flush_failure_keeps_event_for_retry(tmp_path, monkeypatch):
    path = tmp_path / "usage.sqlite3"
    outbox = UsageOutbox(str(path), endpoint="http://d10.test/usage", internal_key="key")
    await outbox.initialize()
    await outbox.enqueue(_event())

    class _FailingClient:
        async def __aenter__(self):
            raise httpx.ConnectError("offline")

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(
        "agent.d10.usage_outbox.httpx.AsyncClient",
        lambda **_kwargs: _FailingClient(),
    )
    assert await outbox.flush_once() == 0
    assert await outbox.pending_count() == 1
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT attempts, last_error FROM d10_usage_outbox"
        ).fetchone()
    assert row[0] == 1
    assert "offline" in row[1]
