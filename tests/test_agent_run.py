"""Streaming tests for POST /agent/run."""

import json

import pytest

from tests.conftest import TEST_KEY

_HEADERS = {"x-internal-key": TEST_KEY}
_BODY = {"lab_id": "lab-xyz", "user_id": "u-123", "question": "how many cases today?"}


def _parse_ndjson(resp) -> list[dict]:
    return [json.loads(line) for line in resp.text.strip().splitlines() if line]


def test_run_yields_done_event(client, monkeypatch):
    import server

    async def _mock(**kwargs):
        yield {"type": "status", "step": "thinking"}
        yield {"type": "delta", "text": "5 cases received today."}
        yield {"type": "done"}

    monkeypatch.setattr(server, "run_turn", _mock)

    resp = client.post("/agent/run", json=_BODY, headers=_HEADERS)
    assert resp.status_code == 200
    events = _parse_ndjson(resp)
    types = [e["type"] for e in events]
    assert "done" in types
    assert "delta" in types


def test_run_forwards_error_event(client, monkeypatch):
    import server

    async def _mock(**kwargs):
        yield {"type": "error", "code": "AGENT_FAILED", "message": "boom"}

    monkeypatch.setattr(server, "run_turn", _mock)

    resp = client.post("/agent/run", json=_BODY, headers=_HEADERS)
    assert resp.status_code == 200  # errors come as stream events, not HTTP errors
    events = _parse_ndjson(resp)
    assert any(e["type"] == "error" for e in events)


def test_run_passes_correct_lab_and_user(client, monkeypatch):
    import server

    received: dict = {}

    async def _mock(*, lab_id, user_id, question, history=None):
        received["lab_id"] = lab_id
        received["user_id"] = user_id
        yield {"type": "done"}

    monkeypatch.setattr(server, "run_turn", _mock)

    client.post("/agent/run", json=_BODY, headers=_HEADERS)
    assert received["lab_id"] == "lab-xyz"
    assert received["user_id"] == "u-123"


def test_run_passes_history(client, monkeypatch):
    import server

    received: dict = {}

    async def _mock(*, lab_id, user_id, question, history=None):
        received["history"] = history
        yield {"type": "done"}

    monkeypatch.setattr(server, "run_turn", _mock)

    body = {**_BODY, "history": [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}]}
    client.post("/agent/run", json=body, headers=_HEADERS)
    assert len(received["history"]) == 2


def test_content_type_is_ndjson(client, monkeypatch):
    import server

    async def _mock(**kwargs):
        yield {"type": "done"}

    monkeypatch.setattr(server, "run_turn", _mock)

    resp = client.post("/agent/run", json=_BODY, headers=_HEADERS)
    assert "ndjson" in resp.headers.get("content-type", "")
