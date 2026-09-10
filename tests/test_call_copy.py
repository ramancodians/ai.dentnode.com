"""Tests for the D10 appointment-reminder call-copy agent.

The bulk of these guard `render_when`, because that is the function standing
between a patient and a phone call naming the wrong day. It is pure and
table-driven, so it can be asserted exactly rather than approximately.
"""

from datetime import datetime, timezone as dt_timezone

import pytest

from agent import call_copy
from agent.call_copy import (
    CALL_COPY_MODEL,
    SUPPORTED_LANGUAGES,
    CallCopy,
    CallCopyError,
    UnsupportedAppointmentTime,
    UnsupportedLanguage,
    generate_call_copy,
    parse_appointment,
    render_when,
)
from agent.openrouter import ChatResult


# ── The deterministic renderer ──────────────────────────────────────────────


def test_no_language_ever_emits_a_digit():
    """A digit reaching a TTS engine is read wrong or read as a numeral.

    Sweeps every hour and every minute in all languages, which is the only way
    to be sure no table gap falls through to a bare int.
    """
    for language in SUPPORTED_LANGUAGES:
        for hour in range(24):
            for minute in range(60):
                spoken = render_when(datetime(2026, 9, 17, hour, minute), language)
                assert not any(ch.isdigit() for ch in spoken), (
                    f"{language} {hour}:{minute:02d} -> {spoken}"
                )


@pytest.mark.parametrize(
    "language,expected",
    [
        ("hi", "गुरुवार, सत्रह सितंबर दो हज़ार छब्बीस को शाम साढ़े चार बजे"),
        ("mr", "गुरुवार, सतरा सप्टेंबर दोन हजार सव्वीस रोजी संध्याकाळी साडेचार वाजता"),
        (
            "kn",
            "ಗುರುವಾರ, ಹದಿನೇಳು ಸೆಪ್ಟೆಂಬರ್ ಎರಡು ಸಾವಿರ ಇಪ್ಪತ್ತಾರು ರಂದು ಸಂಜೆ ನಾಲ್ಕು ಗಂಟೆ ಮೂವತ್ತು ನಿಮಿಷಕ್ಕೆ",
        ),
        (
            "en",
            "Thursday, the seventeenth of September, two thousand twenty-six, "
            "at half past four in the evening",
        ),
    ],
)
def test_known_datetime_renders_exactly(language, expected):
    assert render_when(datetime(2026, 9, 17, 16, 30), language) == expected


@pytest.mark.parametrize(
    "hour,minute,fragment",
    [
        (16, 0, "चार बजे"),
        (16, 15, "सवा चार बजे"),
        (16, 30, "साढ़े चार बजे"),
        (16, 45, "पौने पाँच बजे"),        # quarter TO the next hour
        (13, 30, "डेढ़ बजे"),              # 1:30 has its own word
        (14, 30, "ढाई बजे"),               # so does 2:30
        (9, 20, "नौ बजकर बीस मिनट पर"),   # off-quarter falls back to minutes
    ],
)
def test_hindi_speaks_clock_times_idiomatically(hour, minute, fragment):
    assert fragment in render_when(datetime(2026, 9, 17, hour, minute), "hi")


@pytest.mark.parametrize(
    "hour,minute,fragment",
    [
        (16, 30, "साडेचार वाजता"),   # Marathi joins साडे to the hour
        (13, 30, "दीड वाजता"),
        (14, 30, "अडीच वाजता"),
        (16, 45, "पावणे पाच वाजता"),
    ],
)
def test_marathi_speaks_clock_times_idiomatically(hour, minute, fragment):
    assert fragment in render_when(datetime(2026, 9, 17, hour, minute), "mr")


@pytest.mark.parametrize(
    "hour,minute,fragment",
    [
        (10, 0, "ಬೆಳಿಗ್ಗೆ ಹತ್ತು ಗಂಟೆಗೆ"),
        (16, 15, "ಸಂಜೆ ನಾಲ್ಕು ಗಂಟೆ ಹದಿನೈದು ನಿಮಿಷಕ್ಕೆ"),
        (22, 45, "ರಾತ್ರಿ ಹತ್ತು ಗಂಟೆ ನಲವತ್ತೈದು ನಿಮಿಷಕ್ಕೆ"),
    ],
)
def test_kannada_speaks_clock_times_without_digits(hour, minute, fragment):
    assert fragment in render_when(datetime(2026, 9, 17, hour, minute), "kn")


@pytest.mark.parametrize(
    "hour,part_hi", [(8, "सुबह"), (13, "दोपहर"), (17, "शाम"), (22, "रात")]
)
def test_daypart_tracks_the_clock(hour, part_hi):
    assert part_hi in render_when(datetime(2026, 9, 17, hour, 0), "hi")


def test_weekday_comes_from_the_date_not_the_caller():
    """17 Sep 2026 is a Thursday; 12 Sep 2026 is a Saturday."""
    assert "गुरुवार" in render_when(datetime(2026, 9, 17, 10, 0), "hi")
    assert "शनिवार" in render_when(datetime(2026, 9, 12, 10, 0), "hi")


def test_year_outside_the_table_is_refused_not_guessed():
    with pytest.raises(UnsupportedAppointmentTime):
        render_when(datetime(2075, 1, 1, 10, 0), "hi")


def test_unknown_language_is_rejected():
    with pytest.raises(UnsupportedLanguage):
        render_when(datetime(2026, 9, 17, 10, 0), "kl")


# ── Appointment parsing ─────────────────────────────────────────────────────


def test_naive_datetime_is_read_as_local_to_the_given_zone():
    parsed = parse_appointment("2026-09-17T16:30:00", "Asia/Kolkata")
    assert (parsed.hour, parsed.minute) == (16, 30)


def test_aware_datetime_is_converted_into_the_given_zone():
    """16:30 UTC is 22:00 IST — a reminder read in UTC would be 5.5h wrong."""
    parsed = parse_appointment("2026-09-17T16:30:00+00:00", "Asia/Kolkata")
    assert (parsed.hour, parsed.minute) == (22, 0)


def test_z_suffix_is_accepted():
    assert parse_appointment("2026-09-17T16:30:00Z", "UTC").hour == 16


def test_legacy_timezone_alias_still_resolves():
    """Chrome on Windows reports Asia/Calcutta; tzdata must cover the alias."""
    assert parse_appointment("2026-09-17T16:30:00", "Asia/Calcutta").hour == 16


@pytest.mark.parametrize("bad", ["not-a-date", "", "17/09/2026"])
def test_unparseable_appointment_is_rejected(bad):
    with pytest.raises(ValueError):
        parse_appointment(bad)


def test_unknown_timezone_is_rejected():
    with pytest.raises(ValueError):
        parse_appointment("2026-09-17T16:30:00", "Mars/Olympus")


# ── Generation, validation and the repair turn ──────────────────────────────


def _chat(text: str) -> ChatResult:
    return ChatResult(
        text=text, model=CALL_COPY_MODEL, usage={"input_tokens": 10},
        cost_usd=0.00001, latency_ms=100,
    )


BASE = dict(
    language="hi",
    appointment_at="2026-09-17T16:30:00",
    patient_name="Sunita Deshmukh",
    clinic_name="Smile Care Dental",
)
WHEN_HI = "गुरुवार, सत्रह सितंबर दो हज़ार छब्बीस को शाम साढ़े चार बजे"
GOOD = (
    "नमस्ते सुनीता देशमुख जी, मैं स्माइल केयर डेंटल से बोल रही हूँ। आपका "
    f"अपॉइंटमेंट {WHEN_HI} है। कृपया आना पक्का करें या हमें वापस कॉल करें।"
)


@pytest.mark.asyncio
async def test_clean_first_answer_is_returned_unrepaired(monkeypatch):
    async def fake(**kwargs):
        return _chat(GOOD)

    monkeypatch.setattr(call_copy, "chat_completion", fake)
    result = await generate_call_copy(**BASE)
    assert isinstance(result, CallCopy)
    assert result.repaired is False
    assert result.spoken_when == WHEN_HI
    assert WHEN_HI in result.text


@pytest.mark.asyncio
async def test_the_pinned_date_phrase_is_handed_to_the_model(monkeypatch):
    """The model must never be asked to compute the date — only to copy it."""
    seen = {}

    async def fake(**kwargs):
        seen["messages"] = kwargs["messages"]
        seen["model"] = kwargs["model"]
        return _chat(GOOD)

    monkeypatch.setattr(call_copy, "chat_completion", fake)
    await generate_call_copy(**BASE)
    assert WHEN_HI in seen["messages"][-1]["content"]
    # The model is a hardcoded constant, never an env lookup.
    assert seen["model"] == CALL_COPY_MODEL


@pytest.mark.asyncio
async def test_a_rewritten_date_triggers_one_repair_turn(monkeypatch):
    """The failure that matters: the model paraphrases the date."""
    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _chat(
                "नमस्ते सुनीता जी, आपका अपॉइंटमेंट गुरुवार, सत्रह सितंबर दो हज़ार "
                "सत्ताईस को शाम साढ़े पाँच बजे है। कृपया आना पक्का करें या कॉल करें।"
            )
        return _chat(GOOD)

    monkeypatch.setattr(call_copy, "chat_completion", fake)
    result = await generate_call_copy(**BASE)
    assert result.repaired is True
    assert WHEN_HI in result.text
    assert len(calls) == 2
    # The corrective turn must say precisely what was wrong.
    assert WHEN_HI in calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_a_stray_digit_triggers_repair(monkeypatch):
    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _chat(GOOD + " कमरा नंबर 4 पर आएँ।")
        return _chat(GOOD)

    monkeypatch.setattr(call_copy, "chat_completion", fake)
    result = await generate_call_copy(**BASE)
    assert result.repaired is True
    assert not any(ch.isdigit() for ch in result.text)


@pytest.mark.asyncio
async def test_copy_that_stays_broken_is_refused_not_spoken(monkeypatch):
    """Better no call than a call naming the wrong day."""

    async def fake(**kwargs):
        return _chat("आपका अपॉइंटमेंट किसी और दिन है। कृपया क्लिनिक को फोन करके पूछ लें।")

    monkeypatch.setattr(call_copy, "chat_completion", fake)
    with pytest.raises(CallCopyError):
        await generate_call_copy(**BASE)


@pytest.mark.asyncio
async def test_repair_cost_and_latency_are_accumulated(monkeypatch):
    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)
        return _chat("too short") if len(calls) == 1 else _chat(GOOD)

    monkeypatch.setattr(call_copy, "chat_completion", fake)
    result = await generate_call_copy(**BASE)
    # Both turns are billed to the lab, so both must reach the ledger.
    assert result.cost_usd == pytest.approx(0.00002)
    assert result.latency_ms == 200


@pytest.mark.asyncio
async def test_model_wrapping_quotes_and_fences_are_stripped(monkeypatch):
    async def fake(**kwargs):
        return _chat(f'"{GOOD}"')

    monkeypatch.setattr(call_copy, "chat_completion", fake)
    result = await generate_call_copy(**BASE)
    assert not result.text.startswith('"')
    assert result.repaired is False


@pytest.mark.asyncio
async def test_gender_is_passed_through_for_verb_inflection(monkeypatch):
    seen = {}

    async def fake(**kwargs):
        seen["content"] = kwargs["messages"][-1]["content"]
        return _chat(GOOD)

    monkeypatch.setattr(call_copy, "chat_completion", fake)
    await generate_call_copy(**BASE, patient_gender="female", caller_gender="male")
    assert "female" in seen["content"] and "male" in seen["content"]


@pytest.mark.asyncio
async def test_absent_optional_facts_are_not_mentioned_to_the_model(monkeypatch):
    """"Invent nothing" only holds if we do not hand over empty labels."""
    seen = {}

    async def fake(**kwargs):
        seen["content"] = kwargs["messages"][-1]["content"]
        return _chat(GOOD)

    monkeypatch.setattr(call_copy, "chat_completion", fake)
    await generate_call_copy(**BASE)
    assert "Doctor:" not in seen["content"]
    assert "Reason for the visit:" not in seen["content"]


@pytest.mark.asyncio
async def test_bad_language_never_reaches_the_model(monkeypatch):
    async def fake(**kwargs):  # pragma: no cover - must not run
        raise AssertionError("model must not be called for an invalid language")

    monkeypatch.setattr(call_copy, "chat_completion", fake)
    with pytest.raises(UnsupportedLanguage):
        await generate_call_copy(**{**BASE, "language": "kl"})


@pytest.mark.asyncio
async def test_missing_required_names_are_rejected(monkeypatch):
    async def fake(**kwargs):  # pragma: no cover - must not run
        raise AssertionError("model must not be called")

    monkeypatch.setattr(call_copy, "chat_completion", fake)
    with pytest.raises(ValueError):
        await generate_call_copy(**{**BASE, "patient_name": "   "})
