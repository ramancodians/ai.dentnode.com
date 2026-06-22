"""Auth tests for POST /agent/run."""

import json

import pytest

from tests.conftest import TEST_KEY

_BODY = {"lab_id": "lab-1", "user_id": "u-1", "question": "hello"}


def test_missing_key_returns_401(client):
    resp = client.post("/agent/run", json=_BODY)
    assert resp.status_code == 401


def test_wrong_key_returns_401(client):
    resp = client.post("/agent/run", json=_BODY, headers={"x-internal-key": "bad-key"})
    assert resp.status_code == 401


def test_correct_key_accepted(client, monkeypatch):
    import server

    async def _mock(**kwargs):
        yield {"type": "done"}

    monkeypatch.setattr(server, "run_turn", _mock)

    resp = client.post("/agent/run", json=_BODY, headers={"x-internal-key": TEST_KEY})
    assert resp.status_code == 200


def test_missing_required_field_returns_422(client):
    resp = client.post(
        "/agent/run",
        json={"lab_id": "lab-1"},  # missing user_id and question
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 422
