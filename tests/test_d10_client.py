"""D10 internal client validation tests."""

import httpx
import pytest

from agent.d10 import client
from agent.d10.context import D10RequestContext


def _context():
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


@pytest.mark.asyncio
async def test_catalog_normalizes_compact_tool_schema(monkeypatch):
    async def _request(*_args, **_kwargs):
        return httpx.Response(
            200,
            json={
                "tools": [
                    {
                        "name": "schedule_reminder",
                        "description": "Schedule it",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ]
            },
        )

    monkeypatch.setattr(client, "_request", _request)
    tools = await client.fetch_tool_catalog(_context())
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "schedule_reminder"


@pytest.mark.asyncio
async def test_tool_call_always_carries_trusted_context(monkeypatch):
    captured = {}

    async def _request(*_args, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(200, json={"success": True, "result": {"job_id": "j1"}})

    monkeypatch.setattr(client, "_request", _request)
    result = await client.call_tool(
        name="generate_warranty_card",
        parameters={"patient_id": "patient-1"},
        context=_context(),
        tool_call_id="call-1",
    )
    assert result == {"job_id": "j1"}
    assert captured["context"]["clinic_id"] == "clinic-1"
    assert "clinic_id" not in captured["parameters"]
