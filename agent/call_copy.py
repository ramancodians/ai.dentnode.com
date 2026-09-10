"""Appointment-reminder call copy for D10 — spoken words, in the patient's language.

D10 places appointment-reminder calls. This agent writes what the caller says:
one short, warm opening in the patient's own language, ready to hand to a TTS
engine or to read off a screen at the front desk.

WHY THE DATE IS NOT WRITTEN BY THE MODEL
----------------------------------------
The whole point of the call is a date and a time. Every model tried on this task
got them wrong when asked to spell them out. `gemma-3-4b` produced "दो हज़ार
छشرين" — Arabic script inside a Devanagari year — and moved the year to 2027 in
Marathi. `gpt-oss-20b` rendered 2026 as "दो हजार साठ और छह" (two thousand sixty
and six) and translated "root canal" literally as "मूल जड़ कैनाल". A reminder
call that names the wrong day is worse than no call at all: the patient misses
the slot and blames the clinic.

So the date and time are rendered here, deterministically, from tables — no model
involvement — and the model is handed the finished phrase with an instruction to
copy it verbatim. `generate_call_copy` then verifies the phrase survived and that
no digit reached the output, and spends one corrective turn if it did. Measured
across seven models, injecting the phrase this way took date accuracy from 0/12
to 10/12 before the verification step runs at all.

Adding a language is a data-only change: add an entry to `_LANGUAGES`. Nothing in
the prompt or the request path is language-specific.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .openrouter import ChatResult, chat_completion

logger = logging.getLogger(__name__)

# ── Model configuration — deliberately hardcoded, NOT read from the env ──────
#
# This is not a secret and not an environment-specific value. The same model must
# write the same copy in dev and in production, or the thing we reviewed is not
# the thing that calls the patient. Keeping it here also keeps it reviewable in a
# diff, which a value set inside a deploy workflow is not.
#
# google/gemma-4-26b-a4b-it was picked by running the real task — a Hindi and a
# Marathi reminder — across seven candidates. It was the only one that produced
# clean, natural, fully-native output in BOTH languages. For the record:
#   llama-3.3-70b   truncated Marathi mid-sentence at 8 words, and cost 3x more
#   mistral-small   Marathi was semantically broken ("I am the opportunity of
#                   the clinic"), and it left names in Latin script in Hindi
#   mistral-nemo    romanised Hindi to Hinglish; answered Marathi in English
#   qwen3-30b       usable, but stiff Marathi honorifics ("महोदयांनो")
#
# THE FREE TIERS DO NOT WORK FOR THIS. Checked against the live catalog, not
# assumed: of 431 models 21 are zero-cost, and every one that could serve this
# task failed to. `google/gemma-4-31b-it:free` and `gemma-4-26b-a4b-it:free`
# returned 429 "temporarily rate-limited upstream" on every attempt across
# several runs. `nvidia/nemotron-3.5-lightning:free` is a reasoning model that
# spent its whole budget narrating a thinking process and never reached an
# answer. `thinkingmachines/inkling-small:free` returns 403 (agentic harnesses
# only). A reminder call that fails because a free tier is busy is a patient who
# does not get reminded, so this path pays — and it pays very little: about
# $0.000047 per call, roughly ₹0.004, or ₹40 for ten thousand calls.
CALL_COPY_MODEL = "google/gemma-4-26b-a4b-it"
CALL_COPY_TEMPERATURE = 0.6
CALL_COPY_TIMEOUT_SECS = 45
CALL_COPY_MAX_TOKENS = 400
# Bounds on the spoken copy. Short enough to stay inside the few seconds a
# patient gives an unknown caller before deciding whether to keep listening.
CALL_COPY_MIN_WORDS = 20
CALL_COPY_MAX_WORDS = 90
# Free-text fields are read aloud to a patient, so they are length-capped to stop
# a malformed record turning into a minute of synthesised speech.
_MAX_FIELD_CHARS = 120


class CallCopyError(RuntimeError):
    """Copy could not be produced to a standard safe to speak to a patient."""


class UnsupportedLanguage(ValueError):
    """Requested language has no rendering tables."""


class UnsupportedAppointmentTime(ValueError):
    """The appointment falls outside what the number tables can render."""


@dataclass
class CallCopy:
    text: str
    language: str
    spoken_when: str
    model: str
    usage: Dict[str, int]
    cost_usd: Optional[float]
    latency_ms: int
    repaired: bool


# ── Language tables ─────────────────────────────────────────────────────────
#
# NUMBERS 1-59 ARE A TABLE, NOT AN ALGORITHM. Hindi and Marathi numerals are
# irregular up to 99 — each has its own form and none is composable from its
# digits — so there is nothing to compute here and a table is the only correct
# shape for it.
#
# NOTE FOR REVIEW: the Marathi forms in the 32-59 range are the ones a native
# speaker should read before this carries real calls. They are the least common
# in appointment slots and the least verifiable from here. Everything below 32,
# and the idiomatic quarter-hour forms, are the well-trodden cases.

_HI_NUM: Tuple[str, ...] = (
    "", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
    "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह",
    "उन्नीस", "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस",
    "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस", "इकतीस", "बत्तीस", "तैंतीस",
    "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस",
    "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस",
    "सैंतालीस", "अड़तालीस", "उनचास", "पचास", "इक्यावन", "बावन", "तिरेपन",
    "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
)

_MR_NUM: Tuple[str, ...] = (
    "", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ", "दहा",
    "अकरा", "बारा", "तेरा", "चौदा", "पंधरा", "सोळा", "सतरा", "अठरा",
    "एकोणीस", "वीस", "एकवीस", "बावीस", "तेवीस", "चोवीस", "पंचवीस", "सव्वीस",
    "सत्तावीस", "अठ्ठावीस", "एकोणतीस", "तीस", "एकतीस", "बत्तीस", "तेहतीस",
    "चौतीस", "पस्तीस", "छत्तीस", "सदतीस", "अडतीस", "एकोणचाळीस", "चाळीस",
    "एक्केचाळीस", "बेचाळीस", "त्रेचाळीस", "चव्वेचाळीस", "पंचेचाळीस",
    "शेहेचाळीस", "सत्तेचाळीस", "अठ्ठेचाळीस", "एकोणपन्नास", "पन्नास",
    "एक्कावन्न", "बावन्न", "त्रेपन्न", "चौपन्न", "पंचावन्न", "छप्पन्न",
    "सत्तावन्न", "अठ्ठावन्न", "एकोणसाठ",
)

_KN_NUM: Tuple[str, ...] = (
    "", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು", "ಹತ್ತು",
    "ಹನ್ನೊಂದು", "ಹನ್ನೆರಡು", "ಹದಿಮೂರು", "ಹದಿನಾಲ್ಕು", "ಹದಿನೈದು", "ಹದಿನಾರು", "ಹದಿನೇಳು",
    "ಹದಿನೆಂಟು", "ಹತ್ತೊಂಬತ್ತು", "ಇಪ್ಪತ್ತು", "ಇಪ್ಪತ್ತೊಂದು", "ಇಪ್ಪತ್ತೆರಡು", "ಇಪ್ಪತ್ತಮೂರು",
    "ಇಪ್ಪತ್ತನಾಲ್ಕು", "ಇಪ್ಪತ್ತೈದು", "ಇಪ್ಪತ್ತಾರು", "ಇಪ್ಪತ್ತೇಳು", "ಇಪ್ಪತ್ತೆಂಟು",
    "ಇಪ್ಪತ್ತೊಂಬತ್ತು", "ಮೂವತ್ತು", "ಮೂವತ್ತೊಂದು", "ಮೂವತ್ತೆರಡು", "ಮೂವತ್ತಮೂರು",
    "ಮೂವತ್ತನಾಲ್ಕು", "ಮೂವತ್ತೈದು", "ಮೂವತ್ತಾರು", "ಮೂವತ್ತೇಳು", "ಮೂವತ್ತೆಂಟು",
    "ಮೂವತ್ತೊಂಬತ್ತು", "ನಲವತ್ತು", "ನಲವತ್ತೊಂದು", "ನಲವತ್ತೆರಡು", "ನಲವತ್ತಮೂರು",
    "ನಲವತ್ತನಾಲ್ಕು", "ನಲವತ್ತೈದು", "ನಲವತ್ತಾರು", "ನಲವತ್ತೇಳು", "ನಲವತ್ತೆಂಟು",
    "ನಲವತ್ತೊಂಬತ್ತು", "ಐವತ್ತು", "ಐವತ್ತೊಂದು", "ಐವತ್ತೆರಡು", "ಐವತ್ತಮೂರು",
    "ಐವತ್ತನಾಲ್ಕು", "ಐವತ್ತೈದು", "ಐವತ್ತಾರು", "ಐವತ್ತೇಳು", "ಐವತ್ತೆಂಟು", "ಐವತ್ತೊಂಬತ್ತು",
)

_EN_NUM: Tuple[str, ...] = (
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "twenty-one", "twenty-two",
    "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven",
    "twenty-eight", "twenty-nine", "thirty", "thirty-one", "thirty-two",
    "thirty-three", "thirty-four", "thirty-five", "thirty-six", "thirty-seven",
    "thirty-eight", "thirty-nine", "forty", "forty-one", "forty-two",
    "forty-three", "forty-four", "forty-five", "forty-six", "forty-seven",
    "forty-eight", "forty-nine", "fifty", "fifty-one", "fifty-two",
    "fifty-three", "fifty-four", "fifty-five", "fifty-six", "fifty-seven",
    "fifty-eight", "fifty-nine",
)

# English says "the twelfth of September", not "the twelve of September". Hindi
# and Marathi use the plain cardinal for a date, so only English needs these.
_EN_ORD: Tuple[str, ...] = (
    "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third",
    "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh",
    "twenty-eighth", "twenty-ninth", "thirtieth", "thirty-first",
)


@dataclass(frozen=True)
class _Language:
    code: str
    english_name: str            # what we call the language to the model
    script_name: str             # named in the prompt so it cannot romanise
    numbers: Sequence[str]
    weekdays: Sequence[str]      # Monday-first, matching datetime.weekday()
    months: Sequence[str]        # index 1-12
    dayparts: Dict[str, str]
    thousand: str
    date_suffix: str             # particle after the date ("को" / "रोजी")
    oclock: str                  # "बजे" / "वाजता"
    minutes_join: Tuple[str, str]  # ("बजकर", "मिनट पर")
    half: str                    # prefix for :30
    quarter_past: str
    quarter_to: str
    half_specials: Dict[int, str]  # 1:30 and 2:30 have their own words
    half_joins: bool             # Marathi writes साडेचार as one word


_LANGUAGES: Dict[str, _Language] = {
    "hi": _Language(
        code="hi", english_name="Hindi", script_name="Devanagari",
        numbers=_HI_NUM,
        weekdays=("सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार",
                  "शनिवार", "रविवार"),
        months=("", "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई",
                "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"),
        dayparts={"morning": "सुबह", "afternoon": "दोपहर",
                  "evening": "शाम", "night": "रात"},
        thousand="हज़ार", date_suffix="को", oclock="बजे",
        minutes_join=("बजकर", "मिनट पर"),
        half="साढ़े", quarter_past="सवा", quarter_to="पौने",
        half_specials={1: "डेढ़", 2: "ढाई"}, half_joins=False,
    ),
    "mr": _Language(
        code="mr", english_name="Marathi", script_name="Devanagari",
        numbers=_MR_NUM,
        weekdays=("सोमवार", "मंगळवार", "बुधवार", "गुरुवार", "शुक्रवार",
                  "शनिवार", "रविवार"),
        months=("", "जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून",
                "जुलै", "ऑगस्ट", "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर"),
        dayparts={"morning": "सकाळी", "afternoon": "दुपारी",
                  "evening": "संध्याकाळी", "night": "रात्री"},
        thousand="हजार", date_suffix="रोजी", oclock="वाजता",
        minutes_join=("वाजून", "मिनिटांनी"),
        half="साडे", quarter_past="सव्वा", quarter_to="पावणे",
        half_specials={1: "दीड", 2: "अडीच"}, half_joins=True,
    ),
    "kn": _Language(
        code="kn", english_name="Kannada", script_name="Kannada",
        numbers=_KN_NUM,
        weekdays=("ಸೋಮವಾರ", "ಮಂಗಳವಾರ", "ಬುಧವಾರ", "ಗುರುವಾರ", "ಶುಕ್ರವಾರ",
                  "ಶನಿವಾರ", "ಭಾನುವಾರ"),
        months=("", "ಜನವರಿ", "ಫೆಬ್ರವರಿ", "ಮಾರ್ಚ್", "ಏಪ್ರಿಲ್", "ಮೇ", "ಜೂನ್",
                "ಜುಲೈ", "ಆಗಸ್ಟ್", "ಸೆಪ್ಟೆಂಬರ್", "ಅಕ್ಟೋಬರ್", "ನವೆಂಬರ್", "ಡಿಸೆಂಬರ್"),
        dayparts={"morning": "ಬೆಳಿಗ್ಗೆ", "afternoon": "ಮಧ್ಯಾಹ್ನ",
                  "evening": "ಸಂಜೆ", "night": "ರಾತ್ರಿ"},
        thousand="ಸಾವಿರ", date_suffix="ರಂದು", oclock="ಗಂಟೆಗೆ",
        minutes_join=("ಗಂಟೆ", "ನಿಮಿಷಕ್ಕೆ"),
        half="", quarter_past="", quarter_to="",
        half_specials={}, half_joins=False,
    ),
    "en": _Language(
        code="en", english_name="English", script_name="Latin",
        numbers=_EN_NUM,
        weekdays=("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                  "Saturday", "Sunday"),
        months=("", "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November",
                "December"),
        dayparts={"morning": "in the morning", "afternoon": "in the afternoon",
                  "evening": "in the evening", "night": "at night"},
        thousand="thousand", date_suffix="", oclock="o'clock",
        minutes_join=("", ""),
        half="half past", quarter_past="quarter past", quarter_to="quarter to",
        half_specials={}, half_joins=False,
    ),
}

SUPPORTED_LANGUAGES: Tuple[str, ...] = tuple(_LANGUAGES)


def _daypart(hour24: int) -> str:
    if 4 <= hour24 < 12:
        return "morning"
    if 12 <= hour24 < 16:
        return "afternoon"
    if 16 <= hour24 < 19:
        return "evening"
    return "night"


def _year_words(lang: _Language, year: int) -> str:
    """Render a year as speech. 2026 -> "दो हज़ार छब्बीस"."""
    if not 2000 <= year <= 2059:
        # Composing above 59 needs number words this table does not carry, and an
        # appointment outside that window is a data error, not a use case.
        raise UnsupportedAppointmentTime(f"Cannot render year {year} as words")
    remainder = year - 2000
    thousands = lang.numbers[2]
    if remainder == 0:
        return f"{thousands} {lang.thousand}"
    return f"{thousands} {lang.thousand} {lang.numbers[remainder]}"


def _time_words(lang: _Language, hour24: int, minute: int) -> str:
    """Render a clock time the way it is spoken, not digit by digit."""
    if not 0 <= minute <= 59:
        raise UnsupportedAppointmentTime(f"Invalid minute {minute}")
    daypart = lang.dayparts[_daypart(hour24)]
    h12 = hour24 % 12 or 12
    next_h12 = (hour24 + 1) % 12 or 12

    if lang.code == "en":
        if minute == 0:
            body = f"{lang.numbers[h12]} {lang.oclock}"
        elif minute == 15:
            body = f"{lang.quarter_past} {lang.numbers[h12]}"
        elif minute == 30:
            body = f"{lang.half} {lang.numbers[h12]}"
        elif minute == 45:
            body = f"{lang.quarter_to} {lang.numbers[next_h12]}"
        else:
            body = f"{lang.numbers[h12]} {lang.numbers[minute]}"
        return f"{body} {daypart}"

    # Kannada clock expressions stay explicit. This is less colloquial than
    # regional quarter-hour idioms, but it is unambiguous and safe for every
    # appointment minute that the API accepts.
    if lang.code == "kn":
        if minute == 0:
            body = f"{lang.numbers[h12]} {lang.oclock}"
        else:
            join_a, join_b = lang.minutes_join
            body = f"{lang.numbers[h12]} {join_a} {lang.numbers[minute]} {join_b}"
        return f"{daypart} {body}"

    # Indic: the idiomatic quarter forms are what a person actually says. A
    # receptionist says "साडेचार", never "चार बजकर तीस मिनट".
    if minute == 0:
        body = f"{lang.numbers[h12]} {lang.oclock}"
    elif minute == 15:
        body = f"{lang.quarter_past} {lang.numbers[h12]} {lang.oclock}"
    elif minute == 30:
        special = lang.half_specials.get(h12)
        if special:
            body = f"{special} {lang.oclock}"
        elif lang.half_joins:
            body = f"{lang.half}{lang.numbers[h12]} {lang.oclock}"
        else:
            body = f"{lang.half} {lang.numbers[h12]} {lang.oclock}"
    elif minute == 45:
        body = f"{lang.quarter_to} {lang.numbers[next_h12]} {lang.oclock}"
    else:
        join_a, join_b = lang.minutes_join
        body = f"{lang.numbers[h12]} {join_a} {lang.numbers[minute]} {join_b}"
    return f"{daypart} {body}"


def render_when(when: datetime, language: str) -> str:
    """Render a datetime as one spoken phrase. Never contains a digit.

    This is the function that makes the feature safe. It is pure, table-driven
    and unit-tested; the model never computes any part of its output.
    """
    lang = _LANGUAGES.get(language)
    if lang is None:
        raise UnsupportedLanguage(
            f"Unsupported language {language!r}. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    weekday = lang.weekdays[when.weekday()]
    month = lang.months[when.month]
    year = _year_words(lang, when.year)
    time_part = _time_words(lang, when.hour, when.minute)

    if lang.code == "en":
        day = _EN_ORD[when.day]
        return f"{weekday}, the {day} of {month}, {year}, at {time_part}"
    day = lang.numbers[when.day]
    return f"{weekday}, {day} {month} {year} {lang.date_suffix} {time_part}"


# ── Prompt ──────────────────────────────────────────────────────────────────

_SYSTEM = """You write the spoken opening of a warm, brief appointment-reminder \
phone call made by a dental clinic's front desk.

Output ONLY the words to be spoken. No markdown, no emoji, no quotation marks, \
no stage directions, no preamble, no explanation, no sign-off label.

Write entirely in {language} using {script} script. Never romanise, never \
transliterate, and use no English except for proper nouns with no natural form \
in {language}.

THE APPOINTMENT DATE AND TIME IS GIVEN TO YOU ALREADY WRITTEN OUT IN WORDS. \
Copy that phrase EXACTLY, character for character, into your sentence. Never \
re-express it, translate it, recompute it, reorder it, abbreviate it or round \
it. Never write a digit anywhere in your output.

Sound like a person on the phone, not like a letter: natural spoken register, \
the particles and contractions a real receptionist uses. Be warm but brief — \
the patient decides within seconds whether to keep listening.

Cover, in this order: a greeting to the patient by name, who is calling and \
from which clinic, that this is a reminder, the doctor, the reason for the \
visit, and the date and time. Close by asking them to confirm, or to call back \
if they need to change it.

Invent nothing. Use only the facts given. If a fact is absent, leave it out \
rather than filling it in. Between {min_words} and {max_words} words."""


def _clean(value: Optional[str]) -> str:
    """Collapse whitespace and cap length — these strings get spoken aloud."""
    if not value:
        return ""
    return " ".join(str(value).split())[:_MAX_FIELD_CHARS]


def build_messages(
    *,
    language: str,
    spoken_when: str,
    patient_name: str,
    clinic_name: str,
    doctor_name: str = "",
    reason: str = "",
    caller_name: str = "",
    patient_gender: str = "",
    caller_gender: str = "",
    notes: str = "",
) -> List[Dict[str, str]]:
    lang = _LANGUAGES[language]
    system = _SYSTEM.format(
        language=lang.english_name,
        script=lang.script_name,
        min_words=CALL_COPY_MIN_WORDS,
        max_words=CALL_COPY_MAX_WORDS,
    )
    lines = [f"Patient: {patient_name}"]
    # Hindi and Marathi inflect verbs and honorifics for gender. Without this the
    # model guesses, and it guessed inconsistently across the test runs — the
    # same prompt produced both "बोल रही हूँ" and "बोल रहा हूँ".
    if patient_gender:
        lines.append(
            f"Patient's gender (for correct verb and honorific forms): {patient_gender}"
        )
    lines.append(f"Clinic: {clinic_name}")
    if doctor_name:
        lines.append(f"Doctor: {doctor_name}")
    if reason:
        lines.append(f"Reason for the visit: {reason}")
    if caller_name:
        lines.append(f"Caller (the person speaking): {caller_name}")
    if caller_gender:
        lines.append(f"Caller's gender (for correct verb forms): {caller_gender}")
    if notes:
        lines.append(f"Extra instruction for this call: {notes}")
    lines.append(
        f"\nAppointment date and time — copy this phrase verbatim:\n{spoken_when}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


_WRAPPING_QUOTES = "\"'“”‘’「」"


def _tidy(text: str) -> str:
    """Strip the wrappers models add despite being told not to."""
    out = (text or "").strip()
    out = re.sub(r"^```[a-zA-Z]*\n?|```$", "", out).strip()
    # A model that quotes the whole utterance is still usable; one that quotes
    # half of it is not, so only strip a matched pair around the whole string.
    if len(out) >= 2 and out[0] in _WRAPPING_QUOTES and out[-1] in _WRAPPING_QUOTES:
        out = out[1:-1].strip()
    return " ".join(out.split())


def _problems(text: str, spoken_when: str) -> List[str]:
    """Everything wrong with a candidate, phrased as an instruction to fix it."""
    issues: List[str] = []
    if spoken_when not in text:
        issues.append(
            "The appointment date and time phrase was changed. It must appear "
            f"EXACTLY as given, character for character: {spoken_when}"
        )
    if any(ch.isdigit() for ch in text):
        issues.append(
            "The text contains a digit. Every number must be written as words."
        )
    words = len(text.split())
    if words < CALL_COPY_MIN_WORDS:
        issues.append(
            f"Too short ({words} words); write at least {CALL_COPY_MIN_WORDS}."
        )
    elif words > CALL_COPY_MAX_WORDS:
        issues.append(
            f"Too long ({words} words); keep it under {CALL_COPY_MAX_WORDS}."
        )
    return issues


def parse_appointment(when: str, timezone: str = "Asia/Kolkata") -> datetime:
    """Parse an ISO-8601 appointment into an aware datetime in `timezone`.

    A naive string is read as already local to `timezone`; an aware one is
    converted into it. Getting this backwards moves the appointment by hours, so
    it is explicit rather than inferred.
    """
    try:
        parsed = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"appointment_at must be ISO-8601, got {when!r}") from exc
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown timezone {timezone!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


async def generate_call_copy(
    *,
    language: str,
    appointment_at: str,
    patient_name: str,
    clinic_name: str,
    timezone: str = "Asia/Kolkata",
    doctor_name: str = "",
    reason: str = "",
    caller_name: str = "",
    patient_gender: str = "",
    caller_gender: str = "",
    notes: str = "",
) -> CallCopy:
    """Write the spoken opening of one appointment-reminder call.

    Raises:
        UnsupportedLanguage / UnsupportedAppointmentTime / ValueError: bad input.
        OpenRouterError: the model call failed.
        CallCopyError: no speakable copy after one corrective turn.
    """
    if language not in _LANGUAGES:
        raise UnsupportedLanguage(
            f"Unsupported language {language!r}. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    patient_name = _clean(patient_name)
    clinic_name = _clean(clinic_name)
    if not patient_name or not clinic_name:
        raise ValueError("patient_name and clinic_name are required")

    when = parse_appointment(appointment_at, timezone)
    spoken_when = render_when(when, language)

    messages = build_messages(
        language=language,
        spoken_when=spoken_when,
        patient_name=patient_name,
        clinic_name=clinic_name,
        doctor_name=_clean(doctor_name),
        reason=_clean(reason),
        caller_name=_clean(caller_name),
        patient_gender=_clean(patient_gender),
        caller_gender=_clean(caller_gender),
        notes=_clean(notes),
    )

    result: ChatResult = await chat_completion(
        messages=messages,
        model=CALL_COPY_MODEL,
        temperature=CALL_COPY_TEMPERATURE,
        max_tokens=CALL_COPY_MAX_TOKENS,
        timeout_secs=CALL_COPY_TIMEOUT_SECS,
    )
    text = _tidy(result.text)
    issues = _problems(text, spoken_when)
    repaired = False
    total_latency = result.latency_ms
    usage = dict(result.usage)
    cost = result.cost_usd

    if issues:
        # One corrective turn. Cheaper than failing the call, and the failure
        # modes here — a rewritten date, a stray digit — are exactly the kind a
        # model fixes when told precisely what is wrong.
        logger.warning(
            "Call copy needed repair",
            extra={"language": language, "issues": issues},
        )
        repair = await chat_completion(
            messages=messages
            + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "Rewrite it, fixing exactly these problems and "
                    "changing nothing else:\n- " + "\n- ".join(issues),
                },
            ],
            model=CALL_COPY_MODEL,
            temperature=CALL_COPY_TEMPERATURE,
            max_tokens=CALL_COPY_MAX_TOKENS,
            timeout_secs=CALL_COPY_TIMEOUT_SECS,
        )
        repaired = True
        total_latency += repair.latency_ms
        for key, value in repair.usage.items():
            usage[key] = usage.get(key, 0) + value
        if repair.cost_usd is not None:
            cost = (cost or 0.0) + repair.cost_usd
        candidate = _tidy(repair.text)
        remaining = _problems(candidate, spoken_when)
        if remaining:
            # Refuse rather than speak a wrong date to a patient.
            raise CallCopyError(
                "Model could not produce speakable copy: " + "; ".join(remaining)
            )
        text = candidate

    return CallCopy(
        text=text,
        language=language,
        spoken_when=spoken_when,
        model=result.model,
        usage=usage,
        cost_usd=cost,
        latency_ms=total_latency,
        repaired=repaired,
    )
