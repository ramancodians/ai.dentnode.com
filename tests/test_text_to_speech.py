"""Tests for the Text-to-Speech agent.

The upstream call is an SSE stream of base64 PCM frames, so the fake below
speaks httpx's streaming protocol rather than returning a JSON body. Everything
that matters about this feature — the WAV header, the streaming-vs-buffered
split, metering, and failing before the response commits — depends on that
streaming shape, so mocking at the `httpx.AsyncClient.stream` seam is the only
place the tests are worth writing.
"""

import base64
import json
import struct

import httpx
import pytest

from agent import text_to_speech as tts
from agent.openrouter import OpenRouterError
from tests.conftest import D10_TEST_KEY, TEST_KEY

# 24 kHz mono 16-bit = 48000 bytes per second, so this chunk is exactly half a
# second and the two-chunk default stream below is exactly one.
_PCM_CHUNK = b"\x01\x02" * 12000


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}"


def _audio_event(data: bytes = b"", transcript: str = "") -> dict:
    audio = {}
    if data:
        audio["data"] = base64.b64encode(data).decode()
    if transcript:
        audio["transcript"] = transcript
    return {"choices": [{"delta": {"audio": audio}}]}


def _usage_event(cost: float = 0.000245) -> dict:
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": 48,
            "completion_tokens": 90,
            "total_tokens": 138,
            "cost": cost,
            "completion_tokens_details": {"audio_tokens": 69},
        },
    }


def _default_lines():
    return [
        ": OPENROUTER PROCESSING",
        "",
        _sse(_audio_event(transcript="Your crown case ")),
        _sse(_audio_event(_PCM_CHUNK, "is ready.")),
        _sse(_audio_event(_PCM_CHUNK)),
        _sse(_usage_event()),
        "data: [DONE]",
    ]


class _FakeStream:
    """Async context manager standing in for `httpx.AsyncClient.stream(...)`."""

    def __init__(self, lines, status_code=200, body=b"", die_after=None):
        self._lines = lines
        self.status_code = status_code
        self._body = body
        self._die_after = die_after

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aread(self):
        return self._body

    async def aiter_lines(self):
        for i, line in enumerate(self._lines):
            if self._die_after is not None and i == self._die_after:
                # A dropped connection partway through, which is what a
                # truncated audio file actually looks like in production.
                raise httpx.ReadError("connection reset")
            yield line


@pytest.fixture
def fake_openrouter(monkeypatch):
    """Patch the streaming call. Returns a dict the test can inspect/steer."""
    state = {
        "lines": _default_lines(),
        "status_code": 200,
        "body": b"",
        "die_after": None,
        "request": None,
    }

    def _stream(self, method, url, *, json=None, headers=None, **kwargs):
        state["request"] = {"method": method, "url": url, "json": json, "headers": headers}
        return _FakeStream(
            state["lines"], state["status_code"], state["body"], state["die_after"]
        )

    monkeypatch.setattr(httpx.AsyncClient, "stream", _stream)
    return state


@pytest.fixture(autouse=True)
def _no_metering(monkeypatch):
    """Metering is fire-and-forget against Node; record it instead of sending.

    The stub is deliberately a *sync* function returning a throwaway coroutine.
    The endpoint hands `report_usage(...)` to `_fire_and_forget`, which wraps it
    in `asyncio.create_task` — so an async stub would only record once that task
    got scheduled, which is not guaranteed to happen before the assertion runs.
    Recording at call time instead makes these assertions deterministic.
    """
    calls = []

    async def _noop():
        return None

    def _report(**kwargs):
        calls.append(kwargs)
        return _noop()

    monkeypatch.setattr("server.report_usage", _report)
    return calls


@pytest.fixture
def metering(_no_metering):
    return _no_metering


# ── WAV header ──────────────────────────────────────────────────────────────


def test_wav_header_describes_the_pcm_we_actually_receive():
    header = tts.wav_header(48000)
    assert len(header) == tts.WAV_HEADER_BYTES
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    (channels, rate, byte_rate, align, bits) = struct.unpack("<HIIHH", header[22:36])
    assert channels == 1
    assert rate == 24000
    assert bits == 16
    # A wrong byte_rate is the bug that plays audio at the wrong speed, and it
    # is silent — nothing errors, the voice just sounds sped up.
    assert byte_rate == 24000 * 2
    assert align == 2
    assert struct.unpack("<I", header[4:8])[0] == 36 + 48000
    assert struct.unpack("<I", header[40:44])[0] == 48000


def test_wav_header_uses_the_streaming_sentinel_when_length_is_unknown():
    header = tts.wav_header(None)
    assert struct.unpack("<I", header[4:8])[0] == 0xFFFFFFFF
    assert struct.unpack("<I", header[40:44])[0] == 0xFFFFFFFF


# ── validation ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,voice,fmt",
    [
        ("   ", "alloy", "wav"),
        ("hello", "morgan-freeman", "wav"),
        ("hello", "alloy", "mp3"),
    ],
)
def test_validate_request_rejects_bad_input(text, voice, fmt):
    with pytest.raises(ValueError):
        tts.validate_request(text=text, voice=voice, audio_format=fmt)


def test_validate_request_enforces_the_character_cap(monkeypatch):
    monkeypatch.setattr(tts.settings, "text_to_speech_max_chars", 10)
    with pytest.raises(ValueError, match="the limit is 10"):
        tts.validate_request(text="x" * 11, voice="alloy", audio_format="wav")


# ── synthesis ───────────────────────────────────────────────────────────────


async def test_collect_returns_a_wav_with_an_exact_size_header(fake_openrouter):
    audio, meta = await tts.synthesize_speech(
        text="Your crown case is ready.", voice="alloy", audio_format="wav"
    )
    pcm_len = len(_PCM_CHUNK) * 2
    assert len(audio) == tts.WAV_HEADER_BYTES + pcm_len
    assert audio[:4] == b"RIFF"
    # The sentinel from the streaming header must be gone in the buffered path.
    assert struct.unpack("<I", audio[40:44])[0] == pcm_len
    assert audio[tts.WAV_HEADER_BYTES:] == _PCM_CHUNK * 2
    assert meta.duration_ms == 1000
    assert meta.transcript == "Your crown case is ready."
    assert meta.cost_usd == pytest.approx(0.000245)
    assert meta.cost_source == "openrouter"
    assert meta.usage["completion_tokens"] == 90


async def test_pcm_format_skips_the_header(fake_openrouter):
    audio, meta = await tts.synthesize_speech(
        text="hello", voice="sage", audio_format="pcm"
    )
    assert audio == _PCM_CHUNK * 2
    assert meta.voice == "sage"


async def test_request_body_pins_the_openrouter_audio_contract(fake_openrouter):
    await tts.synthesize_speech(text="hello", voice="alloy", audio_format="wav")
    body = fake_openrouter["request"]["json"]
    # Both of these are load-bearing: OpenRouter 400s on audio output without
    # stream:true, and 400s on any container but pcm16 once streaming.
    assert body["stream"] is True
    assert body["audio"]["format"] == "pcm16"
    assert body["modalities"] == ["text", "audio"]
    assert body["usage"] == {"include": True}
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1] == {"role": "user", "content": "hello"}


async def test_open_raises_before_any_byte_when_upstream_errors(fake_openrouter):
    fake_openrouter["status_code"] = 402
    fake_openrouter["body"] = b'{"error":{"message":"Insufficient credits"}}'
    with pytest.raises(OpenRouterError, match="402"):
        await tts.SpeechStream(
            text="hello", voice="alloy", audio_format="wav"
        ).open()


async def test_open_raises_when_the_stream_carries_no_audio(fake_openrouter):
    # A chat model asked for audio can still answer in text only.
    fake_openrouter["lines"] = [
        _sse({"choices": [{"delta": {"content": "I cannot do that."}}]}),
        "data: [DONE]",
    ]
    with pytest.raises(OpenRouterError, match="no audio"):
        await tts.SpeechStream(
            text="hello", voice="alloy", audio_format="wav"
        ).open()


# ── endpoint ────────────────────────────────────────────────────────────────


def test_endpoint_requires_an_internal_key(client, fake_openrouter):
    resp = client.post("/text-to-speech", json={"text": "hello"})
    assert resp.status_code == 401


def test_endpoint_accepts_the_d10_key_too(client, fake_openrouter):
    resp = client.post(
        "/text-to-speech",
        json={"text": "hello"},
        headers={"x-d10-internal-key": D10_TEST_KEY},
    )
    assert resp.status_code == 200


def test_streaming_response_is_a_playable_wav(client, fake_openrouter, metering):
    resp = client.post(
        "/text-to-speech",
        json={"text": "Your crown case is ready.", "voice": "alloy"},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.headers["x-tts-sample-rate"] == "24000"
    assert resp.content[:4] == b"RIFF"
    assert resp.content[tts.WAV_HEADER_BYTES:] == _PCM_CHUNK * 2
    # Metering runs in the generator's finally, so by the time the body is
    # complete the event must already have been queued.
    assert metering[-1]["feature"] == "text_to_speech"
    assert metering[-1]["status"] == "ok"
    assert metering[-1]["cost"] == pytest.approx(0.000245)
    assert metering[-1]["meta"]["audio_ms"] == 1000


def test_buffered_response_carries_the_transcript_and_cost(client, fake_openrouter):
    resp = client.post(
        "/text-to-speech",
        json={"text": "Your crown case is ready.", "stream": False},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 200
    assert resp.headers["content-length"] == str(len(resp.content))
    transcript = base64.b64decode(resp.headers["x-tts-transcript-b64"]).decode()
    assert transcript == "Your crown case is ready."
    assert float(resp.headers["x-tts-cost-usd"]) == pytest.approx(0.000245)
    assert resp.headers["x-tts-audio-ms"] == "1000"


def test_transcript_header_survives_a_non_latin1_script(client, fake_openrouter):
    fake_openrouter["lines"] = [
        _sse(_audio_event(_PCM_CHUNK, "आपका केस तैयार है")),
        _sse(_usage_event()),
        "data: [DONE]",
    ]
    resp = client.post(
        "/text-to-speech",
        json={"text": "आपका केस तैयार है", "stream": False},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 200
    assert base64.b64decode(resp.headers["x-tts-transcript-b64"]).decode() == (
        "आपका केस तैयार है"
    )


def test_endpoint_rejects_an_unknown_voice(client, fake_openrouter):
    resp = client.post(
        "/text-to-speech",
        json={"text": "hello", "voice": "morgan-freeman"},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 400
    # The service reshapes HTTPException into {"success": false, "error": ...}.
    assert "Unsupported voice" in resp.json()["error"]


def test_upstream_failure_is_a_502_not_a_truncated_file(
    client, fake_openrouter, metering
):
    fake_openrouter["status_code"] = 429
    fake_openrouter["body"] = b"rate limited"
    resp = client.post(
        "/text-to-speech",
        json={"text": "hello"},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 502
    assert metering[-1]["status"] == "error"


def test_buffered_mode_502s_rather_than_returning_a_truncated_file(
    client, fake_openrouter, metering
):
    fake_openrouter["lines"] = [
        _sse(_audio_event(_PCM_CHUNK)),
        _sse(_audio_event(_PCM_CHUNK)),
        _sse(_usage_event()),
        "data: [DONE]",
    ]
    # Connection drops after the first frame — no usage, no [DONE].
    fake_openrouter["die_after"] = 1

    resp = client.post(
        "/text-to-speech",
        json={"text": "hello", "stream": False},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 502
    assert metering[-1]["status"] == "error"


def test_long_transcript_header_is_truncated_not_dropped(client, fake_openrouter):
    # 4000 characters of Devanagari would be ~16 KB of base64 — past what most
    # proxies will forward in one header.
    long_text = "आ" * 3000
    fake_openrouter["lines"] = [
        _sse(_audio_event(_PCM_CHUNK, long_text)),
        _sse(_usage_event()),
        "data: [DONE]",
    ]
    resp = client.post(
        "/text-to-speech",
        json={"text": long_text, "stream": False},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.status_code == 200
    assert resp.headers["x-tts-transcript-truncated"] == "true"
    decoded = base64.b64decode(resp.headers["x-tts-transcript-b64"]).decode()
    assert len(decoded) == 1024
    assert decoded == "आ" * 1024


def test_streaming_response_flags_its_sentinel_sized_header(client, fake_openrouter):
    resp = client.post(
        "/text-to-speech",
        json={"text": "hello"},
        headers={"x-internal-key": TEST_KEY},
    )
    assert resp.headers["x-tts-streaming"] == "chunked"
    # No transcript/cost headers on this path — none of it is known in time.
    assert "x-tts-transcript-b64" not in resp.headers


def test_voices_endpoint_lists_what_the_provider_accepts(client):
    resp = client.get(
        "/text-to-speech/voices", headers={"x-internal-key": TEST_KEY}
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "alloy" in payload["voices"]
    assert payload["formats"] == ["wav", "pcm"]
    assert payload["default_voice"] in payload["voices"]
