"""Tests for agent/node_client.py — HTTP calls, retries, and error handling."""

import asyncio

import httpx
import pytest
import respx

from agent.node_client import NodeToolError, call_tool

_TOOL_URL = "http://node-mock:3000/api/internal/laby-tools/cases_received"

_OK_RESPONSE = {
    "success": True,
    "result": {
        "summary": "5 cases today",
        "columns": ["date", "count"],
        "rows": [["2026-06-22", 5]],
        "chart_hint": None,
        "notes": None,
    },
}


@respx.mock
async def test_successful_call_returns_result():
    respx.post(_TOOL_URL).mock(return_value=httpx.Response(200, json=_OK_RESPONSE))
    result = await call_tool("cases_received", "lab-1", {"range": "today"})
    assert result["summary"] == "5 cases today"
    assert result["columns"] == ["date", "count"]


@respx.mock
async def test_401_raises_auth_error():
    respx.post(_TOOL_URL).mock(return_value=httpx.Response(401))
    with pytest.raises(NodeToolError, match="authentication"):
        await call_tool("cases_received", "lab-1", {})


@respx.mock
async def test_500_raises_node_tool_error():
    respx.post(_TOOL_URL).mock(
        return_value=httpx.Response(500, json={"error": "DB unavailable"})
    )
    with pytest.raises(NodeToolError, match="500"):
        await call_tool("cases_received", "lab-1", {})


@respx.mock
async def test_success_false_body_raises_error():
    respx.post(_TOOL_URL).mock(
        return_value=httpx.Response(200, json={"success": False, "error": "no data"})
    )
    with pytest.raises(NodeToolError, match="no data"):
        await call_tool("cases_received", "lab-1", {})


@respx.mock
async def test_network_error_retries_three_times(monkeypatch):
    async def _noop(_):
        pass

    monkeypatch.setattr("agent.node_client.asyncio.sleep", _noop)

    route = respx.post(_TOOL_URL)
    route.mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(NodeToolError, match="3 attempts"):
        await call_tool("cases_received", "lab-1", {})

    assert route.call_count == 3


@respx.mock
async def test_transient_error_then_success(monkeypatch):
    """Succeeds on the third attempt after two network errors."""

    async def _noop(_):
        pass

    monkeypatch.setattr("agent.node_client.asyncio.sleep", _noop)

    route = respx.post(_TOOL_URL)
    route.mock(
        side_effect=[
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
            httpx.Response(200, json=_OK_RESPONSE),
        ]
    )

    result = await call_tool("cases_received", "lab-1", {})
    assert result["summary"] == "5 cases today"
    assert route.call_count == 3


@respx.mock
async def test_401_is_not_retried(monkeypatch):
    """Auth errors must not be retried — they are permanent."""
    calls = []

    async def _noop(_):
        pass

    monkeypatch.setattr("agent.node_client.asyncio.sleep", _noop)

    route = respx.post(_TOOL_URL)
    route.mock(return_value=httpx.Response(401))

    with pytest.raises(NodeToolError):
        await call_tool("cases_received", "lab-1", {})

    assert route.call_count == 1
