"""Tests for the case-from-image vision endpoint and its parser.

The model call itself is always mocked — these cover the contract Node depends
on: auth, request validation, the response envelope, parse tolerance, and the
rule that metering never breaks the reply.
"""

import pytest

from agent.case_from_image import (
    CaseExtractionError,
    CaseExtractionResult,
    parse_extraction_response,
)
from agent.openrouter import OpenRouterError
from tests.conftest import TEST_KEY

_BODY = {
    "lab_id": "lab-1",
    "image_base64": "aGVsbG8=",
    "mime_type": "image/png",
}

_EXTRACTED = {
    "doctor": {"name": "Dr. Mehta"},
    "patient": {"first_name": "Asha", "last_name": "Rao"},
    "case": {"entry_type": "PHYSICAL"},
    "work": [{"product_name": "Zirconia Crown", "teeth": ["11"], "units": 1}],
    "notes": None,
}


def _ok_result() -> CaseExtractionResult:
    return CaseExtractionResult(
        extracted=_EXTRACTED,
        model="google/gemini-2.5-flash",
        usage={"prompt_tokens": 900, "completion_tokens": 120, "total_tokens": 1020},
        cost_usd=0.00042,
        latency_ms=3100,
    )


@pytest.fixture
def no_metering(monkeypatch):
    """Neutralise the fire-and-forget metering task in every endpoint test."""
    import server

    monkeypatch.setattr(server, "_fire_and_forget", lambda coro: coro.close())


# ─── Parser ───────────────────────────────────────────────────────────────────


def test_parses_plain_json():
    assert parse_extraction_response('{"notes": "hi"}') == {"notes": "hi"}


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"notes": "hi"}\n```',
        '```\n{"notes": "hi"}\n```',
        '  {"notes": "hi"}  ',
        'Here is the case:\n{"notes": "hi"}\nHope that helps.',
    ],
)
def test_parses_wrapped_and_prefixed_json(raw):
    """Fenced, padded, and prose-wrapped replies all still parse."""
    assert parse_extraction_response(raw) == {"notes": "hi"}


@pytest.mark.parametrize("raw", ["", "I could not read this form.", "[1, 2, 3]"])
def test_unparseable_output_raises(raw):
    """A non-object reply must raise, never silently yield a blank case."""
    with pytest.raises(CaseExtractionError):
        parse_extraction_response(raw)


# ─── Auth / validation ────────────────────────────────────────────────────────


def test_missing_key_returns_401(client):
    assert client.post("/case-from-image", json=_BODY).status_code == 401


def test_wrong_key_returns_401(client):
    resp = client.post(
        "/case-from-image", json=_BODY, headers={"x-internal-key": "bad-key"}
    )
    assert resp.status_code == 401


def test_missing_required_field_returns_422(client):
    resp = client.post(
        "/case-from-image",
        json={"lab_id": "lab-1"},  # no image_base64
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 422


# ─── Happy path / failures ────────────────────────────────────────────────────


def test_returns_extracted_payload(client, monkeypatch, no_metering):
    import server

    async def _mock(**kwargs):
        assert kwargs["image_base64"] == "aGVsbG8="
        assert kwargs["mime_type"] == "image/png"
        return _ok_result()

    monkeypatch.setattr(server, "extract_case_from_image", _mock)

    resp = client.post(
        "/case-from-image", json=_BODY, headers={"x-internal-key": TEST_KEY}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["extracted"] == _EXTRACTED
    assert body["model"] == "google/gemini-2.5-flash"


def test_mime_type_defaults_when_omitted(client, monkeypatch, no_metering):
    import server

    seen = {}

    async def _mock(**kwargs):
        seen.update(kwargs)
        return _ok_result()

    monkeypatch.setattr(server, "extract_case_from_image", _mock)

    resp = client.post(
        "/case-from-image",
        json={"lab_id": "lab-1", "image_base64": "aGVsbG8="},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 200
    assert seen["mime_type"] == "image/jpeg"


@pytest.mark.parametrize(
    "exc", [OpenRouterError("upstream 429"), CaseExtractionError("bad json")]
)
def test_model_failures_return_502(client, monkeypatch, no_metering, exc):
    import server

    async def _mock(**kwargs):
        raise exc

    monkeypatch.setattr(server, "extract_case_from_image", _mock)

    resp = client.post(
        "/case-from-image", json=_BODY, headers={"x-internal-key": TEST_KEY}
    )
    assert resp.status_code == 502
    assert resp.json()["success"] is False


def test_metering_failure_does_not_break_response(client, monkeypatch):
    """Metering is best-effort and out-of-band.

    report_usage runs as a fire-and-forget task, so even if it blows up when
    awaited, the caller must still get its extraction.
    """
    import server

    async def _mock(**kwargs):
        return _ok_result()

    async def _boom(**kwargs):
        raise RuntimeError("metering exploded")

    monkeypatch.setattr(server, "extract_case_from_image", _mock)
    monkeypatch.setattr(server, "report_usage", _boom)

    resp = client.post(
        "/case-from-image", json=_BODY, headers={"x-internal-key": TEST_KEY}
    )
    assert resp.status_code == 200
    assert resp.json()["extracted"] == _EXTRACTED
