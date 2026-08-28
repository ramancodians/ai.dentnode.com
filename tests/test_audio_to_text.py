"""Tests for the Audio-to-Text agent: URL guard, format sniff, endpoint auth."""

import io
import wave

import pytest

from agent.audio_fetch import AudioFetchError, sniff_audio_format, validate_url
from agent.audio_to_text import UnsupportedAudioFormat, _split, _validate_format
from tests.conftest import D10_TEST_KEY, TEST_KEY


# ── validate_url (Digital Ocean only) ────────────────────────────────────

def test_validate_url_accepts_digitalocean_origin():
    scheme, host, port, _url = validate_url(
        "https://mybucket.nyc3.digitaloceanspaces.com/audio/note.mp3"
    )
    assert scheme == "https"
    assert host == "mybucket.nyc3.digitaloceanspaces.com"
    assert port == 443


def test_validate_url_accepts_digitalocean_cdn():
    _s, host, _p, _u = validate_url(
        "https://mybucket.nyc3.cdn.digitaloceanspaces.com/audio/note.mp3"
    )
    assert host == "mybucket.nyc3.cdn.digitaloceanspaces.com"


def test_validate_url_rejects_non_digitalocean_host():
    with pytest.raises(AudioFetchError):
        validate_url("https://example.com/audio.mp3")


def test_validate_url_rejects_http_by_default():
    with pytest.raises(AudioFetchError):
        validate_url("http://mybucket.nyc3.digitaloceanspaces.com/audio.mp3")


def test_validate_url_rejects_embedded_credentials():
    with pytest.raises(AudioFetchError):
        validate_url("https://user:pass@mybucket.nyc3.digitaloceanspaces.com/a.mp3")


def test_validate_url_rejects_empty():
    with pytest.raises(AudioFetchError):
        validate_url("")


# ── sniff_audio_format (bytes, not headers) ──────────────────────────────

def _wav_bytes():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)
    return buf.getvalue()


def test_sniff_wav():
    assert sniff_audio_format(_wav_bytes(), "https://x.digitaloceanspaces.com/a", None) == "wav"


def test_sniff_mp3_by_id3():
    data = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 32
    assert sniff_audio_format(data, "https://x.digitaloceanspaces.com/a", None) == "mp3"


def test_sniff_mp3_by_frame_sync():
    data = bytes([0xFF, 0xFB, 0x90, 0x00]) + b"\x00" * 32
    assert sniff_audio_format(data, "https://x.digitaloceanspaces.com/a", None) == "mp3"


def test_sniff_flac():
    assert sniff_audio_format(b"fLaC" + b"\x00" * 8, "https://x.digitaloceanspaces.com/a", None) == "flac"


def test_sniff_ogg():
    assert sniff_audio_format(b"OggS" + b"\x00" * 8, "https://x.digitaloceanspaces.com/a", None) == "ogg"


def test_sniff_webm():
    assert sniff_audio_format(b"\x1aE\xdf\xa3" + b"\x00" * 8, "https://x.digitaloceanspaces.com/a", None) == "webm"


# ── transcription helpers ────────────────────────────────────────────────

def test_split_with_summary_marker():
    transcript, summary = _split("Hello world.\n---SUMMARY---\nNeeds a denture.")
    assert transcript == "Hello world."
    assert summary == "Needs a denture."


def test_split_without_marker():
    transcript, summary = _split("Just a transcript.")
    assert transcript == "Just a transcript."
    assert summary is None


def test_validate_format_rejects_webm():
    with pytest.raises(UnsupportedAudioFormat):
        _validate_format("webm")


def test_validate_format_passes_mp3():
    assert _validate_format("mp3") == "mp3"


# ── endpoint (shared auth + DO-only) ─────────────────────────────────────

_BODY = {"audio_url": "https://b.nyc3.digitaloceanspaces.com/a.wav"}


@pytest.fixture
def stubbed_audio(client, monkeypatch):
    """Stub fetch/transcribe/metering so endpoint tests never touch network."""
    import server
    from agent.audio_fetch import FetchedAudio
    from agent.audio_to_text import AudioToTextResult

    async def _fake_fetch(url):
        return FetchedAudio(url=url, data=b"\x00", format="wav")

    async def _fake_transcribe(**kwargs):
        return AudioToTextResult(
            transcript="hello",
            summary="short" if kwargs.get("summary") else None,
            model="m",
            usage={},
            cost_usd=0.0,
            latency_ms=1,
            audio_format="wav",
            audio_bytes=1,
        )

    monkeypatch.setattr(server, "fetch_audio", _fake_fetch)
    monkeypatch.setattr(server, "transcribe_audio", _fake_transcribe)
    monkeypatch.setattr(server, "report_usage", lambda **kw: None)
    monkeypatch.setattr(server, "_fire_and_forget", lambda coro: None)
    return client


def test_audio_to_text_requires_key(client):
    resp = client.post("/audio-to-text", json=_BODY)
    assert resp.status_code == 401


def test_audio_to_text_rejects_wrong_key(client):
    resp = client.post("/audio-to-text", json=_BODY, headers={"x-internal-key": "wrong"})
    assert resp.status_code == 401


def test_audio_to_text_rejects_non_digitalocean_url(client):
    resp = client.post(
        "/audio-to-text",
        json={"audio_url": "https://example.com/a.mp3"},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 400


def test_audio_to_text_accepts_app_key(stubbed_audio):
    resp = stubbed_audio.post("/audio-to-text", json=_BODY, headers={"x-internal-key": TEST_KEY})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["transcript"] == "hello"
    assert body["summary"] == "short"


def test_audio_to_text_accepts_d10_key(stubbed_audio):
    resp = stubbed_audio.post(
        "/audio-to-text",
        json={**_BODY, "summary": False},
        headers={"x-d10-internal-key": D10_TEST_KEY},
    )
    assert resp.status_code == 200
    assert resp.json()["summary"] is None
