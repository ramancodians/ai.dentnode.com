"""Tests for the two migrated cron endpoints.

/rejected-cases-report — per-lab, vision, hard-fails on unparseable JSON.
/product-update-email  — platform-wide, text, attributed to PLATFORM_LAB_ID.
"""

import pytest

from agent.marketing_copy import build_user_prompt as build_marketing_prompt
from agent.openrouter import ChatResult, OpenRouterError
from agent.rejected_cases import (
    RejectedCasesParseError,
    RejectedCasesResult,
    build_user_prompt,
    parse_report,
)
from tests.conftest import TEST_KEY

_CASES = [
    {
        "entry_id": "e1",
        "case_id": "C-9",
        "doctor_name": "Dr. Mehta",
        "work_types": ["Zirconia Crown"],
        "rejected_at": "2026-07-01",
        "rejection_message": "Margins unclear",
        "fault_owner": "client",
        "image_urls": ["https://example.test/a.jpg"],
    }
]

_REPORT = {
    "summary": "Two rejections this period.",
    "health_score": 82,
    "trend": "STABLE",
    "patterns": [],
    "action_items": [],
    "case_analyses": [],
}


@pytest.fixture
def no_metering(monkeypatch):
    import server

    monkeypatch.setattr(server, "_fire_and_forget", lambda coro: coro.close())


def _sync_capture(into: dict):
    """A report_usage stand-in that records kwargs synchronously.

    Metering is dispatched fire-and-forget, so an async capture body would race
    the response. Constructing the coroutine is the deterministic moment.
    """

    def _capture(**kwargs):
        into.update(kwargs)

        async def _noop():
            return None

        return _noop()

    return _capture


# ─── Rejected cases: prompt + parser ──────────────────────────────────────────


def test_user_prompt_includes_every_case_field():
    text = build_user_prompt(_CASES, "Hillstone Dental")
    assert "Lab: Hillstone Dental" in text
    assert "Total rejected cases: 1" in text
    assert "Entry ID: e1" in text
    assert "Display ID: C-9" in text
    assert "Doctor/Clinic: Dr. Mehta" in text
    assert "Work type(s): Zirconia Crown" in text
    assert "Rejection message: Margins unclear" in text
    assert "Fault attributed to: client" in text
    assert "Screenshots attached: 1" in text


def test_user_prompt_uses_node_fallbacks_for_missing_fields():
    """Node printed N/A / (no message) / unspecified — keep that wording."""
    text = build_user_prompt(
        [
            {
                "entry_id": "e2",
                "case_id": None,
                "doctor_name": "Unknown",
                "work_types": [],
                "rejected_at": "2026-07-02",
                "rejection_message": None,
                "fault_owner": None,
                "image_urls": [],
            }
        ],
        "Lab X",
    )
    assert "Display ID: N/A" in text
    assert "Work type(s): Not specified" in text
    assert "Rejection message: (no message)" in text
    assert "Fault attributed to: unspecified" in text


@pytest.mark.parametrize(
    "raw", ['{"summary": "s"}', '```json\n{"summary": "s"}\n```', 'ok\n{"summary": "s"}']
)
def test_parse_report_variants(raw):
    assert parse_report(raw) == {"summary": "s"}


@pytest.mark.parametrize("raw", ["", "no json", "[1,2]"])
def test_parse_report_raises_on_unusable(raw):
    """Node's bare JSON.parse failed the report; keep it a hard error."""
    with pytest.raises(RejectedCasesParseError):
        parse_report(raw)


# ─── Rejected cases: endpoint ─────────────────────────────────────────────────


def test_rejected_cases_requires_auth(client):
    resp = client.post("/rejected-cases-report", json={"lab_id": "lab-1"})
    assert resp.status_code == 401


def test_rejected_cases_returns_report(client, monkeypatch, no_metering):
    import server

    async def _mock(**kwargs):
        assert kwargs["lab_name"] == "Hillstone Dental"
        assert len(kwargs["images"]) == 1
        return RejectedCasesResult(
            report=_REPORT,
            model="google/gemini-2.5-flash",
            usage={"total_tokens": 9000},
            cost_usd=0.004,
            latency_ms=15000,
        )

    monkeypatch.setattr(server, "generate_rejected_cases_report", _mock)

    resp = client.post(
        "/rejected-cases-report",
        json={
            "lab_id": "lab-1",
            "lab_name": "Hillstone Dental",
            "cases": _CASES,
            "images": [{"image_base64": "aaa", "mime_type": "image/jpeg"}],
        },
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 200
    assert resp.json()["report"] == _REPORT


@pytest.mark.parametrize(
    "exc", [OpenRouterError("boom"), RejectedCasesParseError("bad json")]
)
def test_rejected_cases_failures_return_502(client, monkeypatch, no_metering, exc):
    import server

    async def _mock(**kwargs):
        raise exc

    monkeypatch.setattr(server, "generate_rejected_cases_report", _mock)

    resp = client.post(
        "/rejected-cases-report",
        json={"lab_id": "lab-1", "cases": _CASES},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 502


def test_rejected_cases_meters_against_the_real_lab(client, monkeypatch):
    import server

    captured = {}

    async def _mock(**kwargs):
        return RejectedCasesResult(report=_REPORT, model="m", cost_usd=0.004)

    monkeypatch.setattr(server, "generate_rejected_cases_report", _mock)
    monkeypatch.setattr(server, "report_usage", _sync_capture(captured))

    client.post(
        "/rejected-cases-report",
        json={"lab_id": "lab-1", "cases": _CASES, "images": []},
        headers={"x-internal-key": TEST_KEY},
    )
    assert captured["feature"] == "cron_rejected_cases"
    assert captured["lab_id"] == "lab-1"
    assert captured["meta"] == {"cases": 1, "images": 0}


# ─── Product update email ─────────────────────────────────────────────────────


def test_marketing_prompt_embeds_commits():
    text = build_marketing_prompt([{"hash": "abc", "message": "fix invoices"}])
    assert "fix invoices" in text
    assert "Return ONLY the HTML email content" in text


def test_product_update_requires_auth(client):
    assert client.post("/product-update-email", json={"commits": []}).status_code == 401


def test_product_update_returns_html(client, monkeypatch, no_metering):
    import server

    async def _mock(**kwargs):
        return ChatResult(
            text="<h2>Hello</h2>",
            model="deepseek/deepseek-v4-flash",
            usage={"total_tokens": 1200},
            cost_usd=0.0009,
            latency_ms=2000,
        )

    monkeypatch.setattr(server, "generate_product_update_email", _mock)

    resp = client.post(
        "/product-update-email",
        json={"commits": [{"hash": "abc", "message": "fix"}]},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 200
    assert resp.json()["html"] == "<h2>Hello</h2>"


def test_product_update_meters_to_platform_sentinel(client, monkeypatch):
    """It has no owning lab; spend must still land in the ledger."""
    import server

    captured = {}

    async def _mock(**kwargs):
        return ChatResult(text="<p>hi</p>", model="m", cost_usd=0.0009)

    monkeypatch.setattr(server, "generate_product_update_email", _mock)
    monkeypatch.setattr(server, "report_usage", _sync_capture(captured))

    client.post(
        "/product-update-email",
        json={"commits": []},
        headers={"x-internal-key": TEST_KEY},
    )
    assert captured["feature"] == "cron_product_updates"
    assert captured["lab_id"] == server.PLATFORM_LAB_ID == "__platform__"


def test_product_update_failure_returns_502(client, monkeypatch, no_metering):
    import server

    async def _mock(**kwargs):
        raise OpenRouterError("rate limited")

    monkeypatch.setattr(server, "generate_product_update_email", _mock)

    resp = client.post(
        "/product-update-email", json={"commits": []}, headers={"x-internal-key": TEST_KEY}
    )
    assert resp.status_code == 502
