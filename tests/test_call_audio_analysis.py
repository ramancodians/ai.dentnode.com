"""Contract and security tests for internal call recording analysis."""

from types import SimpleNamespace

import pytest

from agent.audio_to_text import AudioSummaryError, AudioToTextResult
from agent.openrouter import OpenRouterError
from tests.conftest import TEST_KEY


WAV_BYTES = b"RIFF\x04\x00\x00\x00WAVE"
MP3_BYTES = b"ID3\x04\x00\x00\x00\x00\x00\x00"
FLAC_BYTES = b"fLaC\x00\x00\x00\x00"
M4A_BYTES = b"\x00\x00\x00\x18ftypM4A "


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


def _post(client, *, content=WAV_BYTES, content_type="audio/wav", headers=None, **data):
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
        {"audio_bytes": WAV_BYTES, "audio_format": "wav", "summary": True}
    ]


@pytest.mark.parametrize(
    ("mime_type", "expected_format", "content"),
    [
        ("audio/wav", "wav", WAV_BYTES),
        ("audio/x-wav", "wav", WAV_BYTES),
        ("audio/mpeg", "mp3", MP3_BYTES),
        ("audio/flac", "flac", FLAC_BYTES),
        ("audio/mp4", "m4a", M4A_BYTES),
    ],
)
def test_accepts_only_explicit_audio_mime_types(
    stubbed_analysis, mime_type, expected_format, content
):
    client, calls = stubbed_analysis

    response = _post(
        client,
        content=content,
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


def test_rejects_declared_mime_and_audio_signature_mismatch(client):
    response = _post(
        client,
        content=MP3_BYTES,
        content_type="audio/wav",
        headers={"x-internal-key": TEST_KEY},
    )

    assert response.status_code == 415
    assert response.json() == {
        "success": False,
        "error": "Audio content does not match declared type",
    }


def test_rejects_unrecognized_bytes_even_with_allowed_mime(client):
    response = _post(
        client,
        content=b"not audio bytes",
        content_type="audio/wav",
        headers={"x-internal-key": TEST_KEY},
    )

    assert response.status_code == 415
    assert response.json() == {"success": False, "error": "Invalid audio content"}


@pytest.mark.parametrize(
    ("mime_type", "truncated"),
    [
        ("audio/wav", b"RIFF"),
        ("audio/x-wav", b"R"),
        ("audio/mpeg", b"\xff"),
        ("audio/flac", b"fLa"),
        ("audio/mp4", b"\x00" * 7),
    ],
)
def test_truncated_supported_audio_headers_return_sanitized_415(
    client, mime_type, truncated
):
    response = _post(
        client,
        content=truncated,
        content_type=mime_type,
        headers={"x-internal-key": TEST_KEY},
    )

    assert response.status_code == 415
    assert response.json() == {"success": False, "error": "Invalid audio content"}


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


def test_unauthenticated_oversized_body_is_rejected_before_size_processing(
    client, monkeypatch
):
    import server

    monkeypatch.setattr(server.settings, "call_audio_request_max_bytes", 512, raising=False)
    response = _post(client, content=WAV_BYTES + b"x" * 2048)

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_ingress_guard_does_not_read_body_before_authentication(monkeypatch):
    import server

    downstream_called = False
    sent = []

    async def downstream(_scope, _receive, _send):
        nonlocal downstream_called
        downstream_called = True

    async def receive():
        raise AssertionError("unauthenticated request body must not be read")

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/internal/call-audio/analyze",
        "headers": [],
    }
    await server.CallAudioIngressMiddleware(downstream)(scope, receive, send)

    assert downstream_called is False
    assert sent[0]["status"] == 401


@pytest.mark.asyncio
async def test_ingress_guard_caps_chunked_body_without_content_length(monkeypatch):
    import server

    monkeypatch.setattr(server.settings, "call_audio_request_max_bytes", 10)
    downstream_called = False
    sent = []
    incoming = iter(
        [
            {"type": "http.request", "body": b"123456", "more_body": True},
            {"type": "http.request", "body": b"789012", "more_body": False},
        ]
    )

    async def downstream(_scope, _receive, _send):
        nonlocal downstream_called
        downstream_called = True

    async def receive():
        return next(incoming)

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/internal/call-audio/analyze",
        "headers": [(b"x-internal-key", TEST_KEY.encode())],
    }
    await server.CallAudioIngressMiddleware(downstream)(scope, receive, send)

    assert downstream_called is False
    assert sent[0]["status"] == 413


def test_authenticated_total_request_body_cap_rejects_unknown_extra_file(
    client, monkeypatch
):
    import server

    monkeypatch.setattr(server.settings, "call_audio_request_max_bytes", 700, raising=False)
    response = client.post(
        "/internal/call-audio/analyze",
        files={
            "file": ("call.wav", WAV_BYTES, "audio/wav"),
            "extra": ("extra.bin", b"x" * 2048, "application/octet-stream"),
        },
        data={"lab_id": "lab-123", "call_id": "call-456"},
        headers={"x-internal-key": TEST_KEY},
    )

    assert response.status_code == 413
    assert response.json() == {"success": False, "error": "Request body is too large"}


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
