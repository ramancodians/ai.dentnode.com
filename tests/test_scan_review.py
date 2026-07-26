"""Tests for the scan-review vision endpoint, its parser, and content builder.

The multi-image content shape is the whole point of this migration (it replaces
Node's single-image montage workaround), so it is asserted directly.
"""

import json

import pytest

from agent.scan_review import (
    ScanReviewResult,
    build_content,
    parse_ai_response,
)
from agent.openrouter import OpenRouterError
from tests.conftest import TEST_KEY

_VIEWS = [
    {"label": "UPPER ARCH · Buccal", "mime_type": "image/png", "image_base64": "aaa"},
    {"label": "LOWER ARCH · Lingual", "mime_type": "image/png", "image_base64": "bbb"},
]

_BODY = {
    "lab_id": "lab-1",
    "case_context": {"case_id": "C-1"},
    "views": _VIEWS,
}


def _ok_result() -> ScanReviewResult:
    return ScanReviewResult(
        summary="Scans look clean.",
        risk_level="LOW",
        overall_score=92,
        findings=[{"id": "f1", "severity": "INFO"}],
        flags=["clean"],
        raw_response='{"summary": "Scans look clean."}',
        model="google/gemini-2.5-flash",
        parsed=True,
        usage={"prompt_tokens": 4000, "completion_tokens": 300, "total_tokens": 4300},
        cost_usd=0.0021,
        latency_ms=8200,
    )


@pytest.fixture
def no_metering(monkeypatch):
    import server

    monkeypatch.setattr(server, "_fire_and_forget", lambda coro: coro.close())


# ─── Content builder ──────────────────────────────────────────────────────────


def test_every_view_becomes_its_own_image_part():
    """The montage hack is gone: N views must produce N image parts."""
    content = build_content(
        case_context={"case_id": "C-1"},
        views=_VIEWS,
        preview_errors=[],
        today="2026-07-26",
    )
    images = [p for p in content if p["type"] == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"]["url"] == "data:image/png;base64,aaa"
    assert images[1]["image_url"]["url"] == "data:image/png;base64,bbb"


def test_text_comes_first_and_carries_the_ordered_legend():
    """OpenRouter parses best with text before images; labels ride in the legend."""
    content = build_content(
        case_context={"case_id": "C-1"},
        views=_VIEWS,
        preview_errors=[],
        today="2026-07-26",
    )
    assert content[0]["type"] == "text"
    assert all(p["type"] == "image_url" for p in content[1:])

    text = content[0]["text"]
    assert "1. UPPER ARCH · Buccal" in text
    assert "2. LOWER ARCH · Lingual" in text
    assert "2026-07-26" in text
    assert "C-1" in text


def test_missing_mime_type_defaults_to_png():
    content = build_content(
        case_context={},
        views=[{"label": "v", "image_base64": "zzz"}],
        preview_errors=[],
        today="2026-07-26",
    )
    assert content[1]["image_url"]["url"] == "data:image/png;base64,zzz"


def test_no_views_yields_text_only_with_limitation_note():
    content = build_content(
        case_context={"case_id": "C-1"},
        views=[],
        preview_errors=["render failed"],
        today="2026-07-26",
    )
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "No scan previews could be generated" in content[0]["text"]
    assert "render failed" in content[0]["text"]


def test_case_context_with_dates_is_serialisable():
    """Case context comes straight from Prisma and can hold datetimes."""
    from datetime import datetime

    content = build_content(
        case_context={"expected_delivery": datetime(2026, 8, 1)},
        views=[],
        preview_errors=[],
        today="2026-07-26",
    )
    assert "2026-08-01" in content[0]["text"]


# ─── Parser ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        '{"summary": "ok"}',
        '```json\n{"summary": "ok"}\n```',
        'Sure!\n{"summary": "ok"}',
    ],
)
def test_parses_json_variants(raw):
    assert parse_ai_response(raw) == {"summary": "ok"}


@pytest.mark.parametrize("raw", ["", "no json here", "[1,2]"])
def test_unparseable_returns_none(raw):
    """Soft failure: the scan review is an optional add-on, so None not raise."""
    assert parse_ai_response(raw) is None


# ─── Endpoint ─────────────────────────────────────────────────────────────────


def test_missing_key_returns_401(client):
    assert client.post("/scan-review", json=_BODY).status_code == 401


def test_missing_required_field_returns_422(client):
    resp = client.post(
        "/scan-review",
        json={"case_context": {}},  # no lab_id
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 422


def test_returns_full_review_payload(client, monkeypatch, no_metering):
    import server

    async def _mock(**kwargs):
        assert len(kwargs["views"]) == 2
        assert kwargs["today"] == "2026-07-26"
        return _ok_result()

    monkeypatch.setattr(server, "generate_scan_review", _mock)

    resp = client.post(
        "/scan-review",
        json={**_BODY, "today": "2026-07-26"},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["summary"] == "Scans look clean."
    assert body["risk_level"] == "LOW"
    assert body["overall_score"] == 92
    assert body["findings"] == [{"id": "f1", "severity": "INFO"}]
    assert body["flags"] == ["clean"]
    assert body["model"] == "google/gemini-2.5-flash"
    assert body["parsed"] is True


def test_today_defaults_when_omitted(client, monkeypatch, no_metering):
    import server

    seen = {}

    async def _mock(**kwargs):
        seen.update(kwargs)
        return _ok_result()

    monkeypatch.setattr(server, "generate_scan_review", _mock)

    resp = client.post("/scan-review", json=_BODY, headers={"x-internal-key": TEST_KEY})
    assert resp.status_code == 200
    # ISO date, not empty/None.
    assert len(seen["today"]) == 10


def test_empty_views_still_accepted(client, monkeypatch, no_metering):
    """Preview rendering can fail entirely; the review still runs on context."""
    import server

    async def _mock(**kwargs):
        assert kwargs["views"] == []
        return _ok_result()

    monkeypatch.setattr(server, "generate_scan_review", _mock)

    resp = client.post(
        "/scan-review",
        json={"lab_id": "lab-1", "case_context": {}, "views": []},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 200


def test_model_failure_returns_502(client, monkeypatch, no_metering):
    import server

    async def _mock(**kwargs):
        raise OpenRouterError("upstream 503")

    monkeypatch.setattr(server, "generate_scan_review", _mock)

    resp = client.post("/scan-review", json=_BODY, headers={"x-internal-key": TEST_KEY})
    assert resp.status_code == 502
    assert resp.json()["success"] is False


def test_metering_receives_view_count_and_parse_flag(client, monkeypatch):
    """meta lets us later separate cost by payload size and output quality."""
    import server

    captured = {}

    async def _mock(**kwargs):
        return _ok_result()

    # Capture SYNCHRONOUSLY at call time. Metering is dispatched as a
    # fire-and-forget task, so asserting on an async body would race the
    # response and flake; building the coroutine is what happens deterministically.
    def _capture(**kwargs):
        captured.update(kwargs)

        async def _noop():
            return None

        return _noop()

    monkeypatch.setattr(server, "generate_scan_review", _mock)
    monkeypatch.setattr(server, "report_usage", _capture)

    resp = client.post("/scan-review", json=_BODY, headers={"x-internal-key": TEST_KEY})
    assert resp.status_code == 200
    assert captured["feature"] == "scan_review"
    assert captured["lab_id"] == "lab-1"
    assert captured["cost_source"] == "openrouter"
    assert captured["meta"] == {"views": 2, "parsed": True}
