"""Tests for the custom accounting report spec endpoint and its parser.

The model call is always mocked. These cover the contract Node depends on:
auth, request validation, the {success, result} envelope, the retry fields
reaching the prompt, and parse tolerance.
"""

import pytest

from agent.custom_report import (
    CustomReportParseError,
    CustomReportResult,
    build_user_prompt,
    parse_reply,
)
from agent.openrouter import OpenRouterError
from tests.conftest import TEST_KEY

_SPEC = {
    "version": 1,
    "dataset": "invoices",
    "columns": [],
    "groupBy": ["invoiceMonth"],
    "aggregates": [{"fn": "sum", "field": "billed"}],
    "dateRange": {"preset": "this_fy"},
    "chart": {"kind": "line", "x": "invoiceMonth", "y": "sum_billed"},
}

_BODY = {
    "lab_id": "lab-1",
    "user_id": "user-1",
    "prompt": "Month-wise billing this financial year as a chart",
    "today": "2026-10-25",
    "catalog": {"datasets": [{"key": "invoices", "fields": []}]},
}


def _ok_result() -> CustomReportResult:
    return CustomReportResult(
        result={"title": "Month-wise billing", "summary": "Billed per month.", "spec": _SPEC},
        model="deepseek/deepseek-v4-flash",
        usage={"prompt_tokens": 2400, "completion_tokens": 160, "total_tokens": 2560},
        cost_usd=0.0003,
        latency_ms=6000,
    )


@pytest.fixture
def no_metering(monkeypatch):
    import server

    monkeypatch.setattr(server, "_fire_and_forget", lambda coro: coro.close())


# ─── Parser ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        '{"title": "t", "spec": {"dataset": "invoices"}}',
        '```json\n{"title": "t", "spec": {"dataset": "invoices"}}\n```',
        'Here you go: {"title": "t", "spec": {"dataset": "invoices"}} done',
    ],
)
def test_parses_plain_fenced_and_wrapped_json(raw):
    assert parse_reply(raw)["spec"] == {"dataset": "invoices"}


@pytest.mark.parametrize("raw", ["no json here", '{"title": "t"}', '{"spec": "nope"}', "[1, 2]"])
def test_reply_without_a_spec_object_raises(raw):
    with pytest.raises(CustomReportParseError):
        parse_reply(raw)


def test_prompt_carries_current_spec_and_retry_errors():
    text = build_user_prompt(
        prompt="only Bangalore clients",
        catalog={"datasets": []},
        today="2026-10-25",
        current_spec={"dataset": "clients"},
        previous_spec={"dataset": "clients", "columns": ["nope"]},
        errors=['columns: unknown field "nope" in clients'],
    )
    assert "Today: 2026-10-25" in text
    assert "Current spec" in text and '"dataset": "clients"' in text
    assert 'unknown field \\"nope\\"' in text or 'unknown field "nope"' in text
    assert text.index("Request:") < text.index("rejected")


# ─── Endpoint ─────────────────────────────────────────────────────────────────


def test_missing_key_returns_401(client):
    assert client.post("/custom-report-spec", json=_BODY).status_code == 401


def test_missing_prompt_returns_422(client):
    body = {k: v for k, v in _BODY.items() if k != "prompt"}
    resp = client.post("/custom-report-spec", json=body, headers={"x-internal-key": TEST_KEY})
    assert resp.status_code == 422


def test_returns_spec_envelope(client, monkeypatch, no_metering):
    import server

    seen = {}

    async def _mock(**kwargs):
        seen.update(kwargs)
        return _ok_result()

    monkeypatch.setattr(server, "generate_custom_report_spec", _mock)

    resp = client.post(
        "/custom-report-spec",
        json={**_BODY, "previous_spec": {"x": 1}, "errors": ["bad"]},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["result"]["spec"] == _SPEC
    assert seen["prompt"] == _BODY["prompt"]
    assert seen["errors"] == ["bad"]
    assert seen["previous_spec"] == {"x": 1}


@pytest.mark.parametrize("exc", [OpenRouterError("upstream 429"), CustomReportParseError("bad json")])
def test_model_failures_return_502(client, monkeypatch, no_metering, exc):
    import server

    async def _mock(**kwargs):
        raise exc

    monkeypatch.setattr(server, "generate_custom_report_spec", _mock)

    resp = client.post("/custom-report-spec", json=_BODY, headers={"x-internal-key": TEST_KEY})
    assert resp.status_code == 502
