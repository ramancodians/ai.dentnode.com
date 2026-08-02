"""Tests for DentNode image-to-entry normalization and endpoint wiring."""

import pytest

from agent.case_from_image import CaseExtractionError
from agent.image_to_entry import (
    ImageToEntryResult,
    build_entry_payload,
    build_tooth_chart,
)
from agent.openrouter import OpenRouterError
from tests.conftest import TEST_KEY


_EXTRACTED = {
    "doctor": {"name": "Dr Asha Rao", "phone": "9999999999"},
    "patient": {"first_name": "Mira", "last_name": "Sen", "gender": "FEMALE"},
    "case": {
        "case_id": "RX-42",
        "entry_type": "DIGITAL",
        "delivery_date": "2026-08-08",
    },
    "work": [
        {
            "product_name": "Zirconia Crown",
            "teeth": ["11", "21", "36", "36", "99"],
            "shade": "A2",
            "shade_guide": "VITA_3D",
            "units": 3,
            "jaw_type": "BOTH",
            "instructions": "Monolithic, light contact",
        }
    ],
    "notes": "Call before dispatch.",
}


def _result() -> ImageToEntryResult:
    return ImageToEntryResult(
        payload=build_entry_payload(_EXTRACTED),
        model="google/gemini-2.5-flash",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        cost_usd=0.001,
        latency_ms=500,
    )


@pytest.fixture
def no_metering(monkeypatch):
    import server

    monkeypatch.setattr(server, "_fire_and_forget", lambda coro: coro.close())


def test_build_tooth_chart_matches_dentnode_quadrants():
    chart = build_tooth_chart(["11", "21", "36", "99"])
    selected = {
        f"{int(section) + 1}{item['name']}"
        for section, items in chart.items()
        for item in items
        if item["isSelected"]
    }
    assert selected == {"11", "21", "36"}
    assert [item["name"] for item in chart["0"]] == [8, 7, 6, 5, 4, 3, 2, 1]
    assert [item["name"] for item in chart["1"]] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_builds_exact_entry_create_sections_and_resolution_hints():
    result = build_entry_payload(_EXTRACTED)
    payload = result["entry_payload"]

    assert set(payload) == {"entry", "patient", "work"}
    assert payload["entry"]["entry_type"] == "DIGITAL"
    assert payload["entry"]["doctor_id"] is None
    assert payload["entry"]["epxected_delivery_date"] == "2026-08-08"
    assert payload["entry"]["notes"] == "Call before dispatch."
    assert payload["patient"]["first_name"] == "Mira"
    assert payload["patient"]["doctor_id"] is None
    assert payload["work"][0]["productId"] is None
    assert payload["work"][0]["shade_guide"] == "VITA_3D_MASTER"
    assert payload["work"][0]["jaw_type"] == "FULL"
    assert result["detected"]["doctor"]["name"] == "Dr Asha Rao"
    assert result["detected"]["work"][0]["product_name"] == "Zirconia Crown"
    assert result["ready_to_create"] is False
    assert "entry.doctor_id" in result["missing_fields"]
    assert "patient.doctor_id" in result["missing_fields"]
    assert "work[0].productId" in result["missing_fields"]


def test_invalid_model_values_are_safely_normalized():
    result = build_entry_payload(
        {
            "patient": {"first_name": "P", "gender": "unknown", "age": "old"},
            "case": {"entry_type": "paper"},
            "work": [{"teeth": ["48"], "units": 0, "shade_guide": "mystery"}],
        }
    )
    payload = result["entry_payload"]
    assert payload["entry"]["entry_type"] == "PHYSICAL"
    assert "gender" not in payload["patient"]
    assert "age" not in payload["patient"]
    assert payload["work"][0]["total_units"] == 1
    assert payload["work"][0]["jaw_type"] == "LOWER"
    assert payload["work"][0]["shade_guide"] == "VITA_CLASSIC"


_BODY = {"lab_id": "lab-1", "image_base64": "ZmFrZQ==", "mime_type": "image/jpeg"}


def test_endpoint_requires_internal_key(client):
    assert client.post("/image-to-entry", json=_BODY).status_code == 401


def test_endpoint_validates_required_fields(client):
    response = client.post(
        "/image-to-entry",
        json={"lab_id": "lab-1"},
        headers={"x-internal-key": TEST_KEY},
    )
    assert response.status_code == 422


def test_endpoint_returns_payload_without_creating_entry(
    client, monkeypatch, no_metering
):
    import server

    seen = {}

    async def _mock(**kwargs):
        seen.update(kwargs)
        return _result()

    monkeypatch.setattr(server, "image_to_entry", _mock)
    response = client.post(
        "/image-to-entry", json=_BODY, headers={"x-internal-key": TEST_KEY}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["entry_payload"]["entry"]["notes"] == "Call before dispatch."
    assert body["ready_to_create"] is False
    assert body["model"] == "google/gemini-2.5-flash"
    assert seen == {"image_base64": "ZmFrZQ==", "mime_type": "image/jpeg"}


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (CaseExtractionError("bad model JSON"), "bad model JSON"),
        (OpenRouterError("upstream failed"), "Image-to-entry model call failed"),
    ],
)
def test_endpoint_translates_model_errors(
    client, monkeypatch, no_metering, error, message
):
    import server

    async def _mock(**kwargs):
        raise error

    monkeypatch.setattr(server, "image_to_entry", _mock)
    response = client.post(
        "/image-to-entry", json=_BODY, headers={"x-internal-key": TEST_KEY}
    )
    assert response.status_code == 502
    assert response.json() == {"success": False, "error": message}


def test_endpoint_meters_feature_and_payload_state(client, monkeypatch):
    import server

    captured = {}

    async def _mock(**kwargs):
        return _result()

    def _capture(**kwargs):
        captured.update(kwargs)

        async def _noop():
            return None

        return _noop()

    monkeypatch.setattr(server, "image_to_entry", _mock)
    monkeypatch.setattr(server, "report_usage", _capture)
    response = client.post(
        "/image-to-entry", json=_BODY, headers={"x-internal-key": TEST_KEY}
    )
    assert response.status_code == 200
    assert captured["feature"] == "image_to_entry"
    assert captured["lab_id"] == "lab-1"
    assert captured["meta"] == {"work_items": 1, "ready_to_create": False}
