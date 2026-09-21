"""Contract and security tests for internal call recording analysis."""

from types import SimpleNamespace

import pytest

from agent.audio_to_text import AudioSummaryError, AudioToTextResult
from agent.openrouter import OpenRouterError
from tests.conftest import TEST_KEY


def _result(*, transcript="Caller asked about the crown.", summary="Crown discussed."):
    return AudioToTextResult(
        transcript=transcript,
        summary=summary,
        model="test-transcriber",
        usage={"total_tokens": 12},
        cost_usd=0.001,
        latency_ms=9,
        audio_format="wav",
        audio_bytes=12,
        segments=[],
    )


def _result_with_summary_usage():
    result = _result()
    result.summary_model = "test-summarizer"
    result.summary_usage = {
        "prompt_tokens": 21,
        "completion_tokens": 5,
        "total_tokens": 26,
    }
    result.summary_cost_usd = 0.002
    result.summary_latency_ms = 11
    return result


@pytest.fixture
def stubbed_analysis(client, monkeypatch):
    import server

    calls = SimpleNamespace(transcribe=[], usage=[])

    async def fake_transcribe(**kwargs):
        calls.transcribe.append(kwargs)
        return _result()

    def fake_report_usage(**kwargs):
        calls.usage.append(kwargs)

    monkeypatch.setattr(server, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(server, "report_usage", fake_report_usage)
    monkeypatch.setattr(server, "_fire_and_forget", lambda _coro: None)
    return client, calls


def _post(client, *, content=b"RIFFaudio", content_type="audio/wav", headers=None, **data):
    return client.post(
        "/internal/call-audio/analyze",
        files={"file": ("call.wav", content, content_type)},
        data={"lab_id": "lab-123", "call_id": "call-456", **data},
        headers=headers or {},
    )


def test_requires_internal_service_key(client):
    response = _post(client)
    assert response.status_code == 401


def test_rejects_wrong_internal_service_key(client):
    response = _post(client, headers={"x-internal-key": "wrong"})
    assert response.status_code == 401


def test_accepts_upload_and_returns_exact_contract(stubbed_analysis):
    client, calls = stubbed_analysis

    response = _post(client, headers={"x-internal-key": TEST_KEY})

    assert response.status_code == 200
    assert response.json() == {
        "transcript": "Caller asked about the crown.",
        "summary": "Crown discussed.",
    }
    assert calls.transcribe == [
        {"audio_bytes": b"RIFFaudio", "audio_format": "wav", "summary": True}
    ]


@pytest.mark.parametrize(
    ("mime_type", "expected_format"),
    [
        ("audio/wav", "wav"),
        ("audio/x-wav", "wav"),
        ("audio/mpeg", "mp3"),
        ("audio/flac", "flac"),
        ("audio/mp4", "m4a"),
    ],
)
def test_accepts_only_explicit_audio_mime_types(
    stubbed_analysis, mime_type, expected_format
):
    client, calls = stubbed_analysis

    response = _post(
        client,
        content_type=mime_type,
        headers={"x-internal-key": TEST_KEY},
    )

    assert response.status_code == 200
    assert calls.transcribe[-1]["audio_format"] == expected_format


@pytest.mark.parametrize(
    "mime_type",
    ["application/octet-stream", "text/plain", "audio/webm", "image/png"],
)
def test_rejects_unapproved_or_unsupported_mime_types(client, mime_type):
    response = _post(
        client,
        content_type=mime_type,
        headers={"x-internal-key": TEST_KEY},
    )

    assert response.status_code == 415
    assert response.json() == {"success": False, "error": "Unsupported audio type"}


def test_rejects_empty_audio(client):
    response = _post(client, content=b"", headers={"x-internal-key": TEST_KEY})
    assert response.status_code == 400
    assert response.json() == {"success": False, "error": "Audio file is empty"}


def test_enforces_bounded_upload_size(client, monkeypatch):
    import server

    monkeypatch.setattr(server.settings, "audio_to_text_max_bytes", 8)
    response = _post(
        client,
        content=b"123456789",
        headers={"x-internal-key": TEST_KEY},
    )

    assert response.status_code == 413
    assert response.json() == {"success": False, "error": "Audio file is too large"}


def test_reports_usage_with_lab_and_call_id(stubbed_analysis, monkeypatch):
    client, calls = stubbed_analysis
    response = _post(client, headers={"x-internal-key": TEST_KEY})

    assert response.status_code == 200
    assert len(calls.usage) == 1
    assert calls.usage[0]["lab_id"] == "lab-123"
    assert calls.usage[0]["request_id"] == "call-456"
    assert calls.usage[0]["feature"] == "call_audio_analysis"


def test_reports_transcription_and_summary_usage_with_lab_and_call_id(
    client, monkeypatch
):
    import server

    calls = []

    async def fake_transcribe(**_kwargs):
        return _result_with_summary_usage()

    monkeypatch.setattr(server, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(server, "report_usage", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(server, "_fire_and_forget", lambda _coro: None)

    response = _post(client, headers={"x-internal-key": TEST_KEY})

    assert response.status_code == 200
    assert response.json() == {
        "transcript": "Caller asked about the crown.",
        "summary": "Crown discussed.",
    }
    assert [event["meta"]["stage"] for event in calls] == [
        "transcription",
        "summary",
    ]
    assert [event["model"] for event in calls] == [
        "test-transcriber",
        "test-summarizer",
    ]
    assert [event["usage"] for event in calls] == [
        {"total_tokens": 12},
        {"prompt_tokens": 21, "completion_tokens": 5, "total_tokens": 26},
    ]
    assert [event["cost"] for event in calls] == [0.001, 0.002]
    assert all(event["lab_id"] == "lab-123" for event in calls)
    assert all(event["request_id"] == "call-456" for event in calls)


def test_preserves_transcription_metering_when_summary_fails(client, monkeypatch):
    import server

    calls = []
    error = AudioSummaryError(_result(summary=None))

    async def fail_summary(**_kwargs):
        raise error

    monkeypatch.setattr(server, "transcribe_audio", fail_summary)
    monkeypatch.setattr(server, "report_usage", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(server, "_fire_and_forget", lambda _coro: None)

    response = _post(client, headers={"x-internal-key": TEST_KEY})

    assert response.status_code == 502
    assert [(event["meta"]["stage"], event["status"]) for event in calls] == [
        ("transcription", "ok"),
        ("summary", "error"),
    ]
    assert calls[0]["model"] == "test-transcriber"
    assert calls[0]["usage"] == {"total_tokens": 12}
    assert calls[0]["cost"] == 0.001
    assert calls[1]["model"] == server.settings.model
    assert all(event["lab_id"] == "lab-123" for event in calls)
    assert all(event["request_id"] == "call-456" for event in calls)


def test_accepts_existing_internal_id_header_alias(stubbed_analysis):
    client, _calls = stubbed_analysis

    response = _post(client, headers={"x-internal-id": TEST_KEY})

    assert response.status_code == 200


def test_model_failure_is_sanitized_and_does_not_log_sensitive_content(
    client, monkeypatch, caplog
):
    import server

    sensitive = "Patient Jane Doe said secret treatment details"

    async def fail_transcription(**_kwargs):
        raise OpenRouterError(sensitive)

    monkeypatch.setattr(server, "transcribe_audio", fail_transcription)
    monkeypatch.setattr(server, "report_usage", lambda **_kwargs: None)
    monkeypatch.setattr(server, "_fire_and_forget", lambda _coro: None)

    with caplog.at_level("ERROR"):
        response = _post(client, headers={"x-internal-key": TEST_KEY})

    assert response.status_code == 502
    assert response.json() == {
        "success": False,
        "error": "Call audio analysis failed",
    }
    assert sensitive not in caplog.text


def test_contract_has_no_remote_url_input(client):
    response = client.post(
        "/internal/call-audio/analyze",
        data={
            "lab_id": "lab-123",
            "call_id": "call-456",
            "audio_url": "http://169.254.169.254/latest/meta-data/",
        },
        headers={"x-internal-key": TEST_KEY},
    )

    assert response.status_code == 422
