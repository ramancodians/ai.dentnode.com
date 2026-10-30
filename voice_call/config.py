"""Per-call configuration for the LiveKit voice agent.

A call is described by the JSON metadata on its agent dispatch. Every model
choice is checked against an allowlist here, so a dispatcher can pick among
models but never name an arbitrary one. Defaults are the combination that won
the Voice Lab latency matrix on 2026-09-30.

Only the LLM leaves LiveKit: it is called through OpenRouter like every other
LLM call in DentNode. STT, TTS and turn detection run on LiveKit Inference.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agent.voice_agent import REPLY_MODELS
from agent.voice_identity import VOICE_ID as ELEVENLABS_VOICE_ID
from .callbacks import CallCallback

# LiveKit reads the worker's name from LIVEKIT_AGENT_NAME. Setting a name puts
# the worker in explicit-dispatch mode: it joins only rooms that ask for it by
# name, never every room in the project. Local runs use a different name so a
# developer's laptop can never pick up a production call.
AGENT_NAME = os.environ.get("LIVEKIT_AGENT_NAME") or "dentnode-voice"

# Deepgram Nova-3 "multi" code-switches between English and Hindi, which is how
# patients actually talk. It has no Marathi, so Marathi uses Gemini Live.
STT_MODELS = (
    "deepgram/nova-3",
    "deepgram/flux-general-multi",
    "assemblyai/universal-streaming-multilingual",
    "google/gemini-3.5-transcribe-live",
)
TTS_MODELS = (
    "fishaudio/s2.1-pro",
    "elevenlabs/eleven_flash_v2_5",
    "cartesia/sonic-3",
    "inworld/inworld-tts-2",
    "deepgram/aura-2",
)
ELEVENLABS_MODEL = "elevenlabs/eleven_flash_v2_5"
# The voice is pinned in agent/voice_identity.py (female, Shivi V); only the key comes from env.
ELEVENLABS_READY = bool(os.environ.get("ELEVEN_API_KEY"))
DEFAULT_TTS_MODEL = ELEVENLABS_MODEL if ELEVENLABS_READY else TTS_MODELS[0]
LANGUAGES = ("", "en", "hi", "mr")

DEFAULT_OBJECTIVE = (
    "Ask how you can help with the dental clinic. Do not invent patient, appointment, or clinic details."
)
MAX_CALL_SECONDS = 300
# All required, or the call is booked as platform spend: D10 rejects an event
# missing any of them, and one rejected event stalls the whole outbox batch.
USAGE_CONTEXT_KEYS = ("clinic_id", "user_id", "actor_id", "conversation_id",
                      "source_message_id", "correlation_id")


_LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}


def opening_instructions(cfg: "CallConfig") -> str:
    """How the agent opens the call when the dispatcher gave no fixed greeting.

    Nobody has spoken yet, so "mirror the caller's language" has nothing to
    mirror: the configured language decides. The clinic is named only when the
    dispatcher named it — otherwise the model invents one."""
    who = f"on behalf of {cfg.clinic_name}" if cfg.clinic_name else (
        "on behalf of their dental clinic (do not name the clinic)")
    language = _LANGUAGE_NAMES.get(cfg.language)
    speak = f" Speak in {language}." if language and language != "English" else ""
    return (f"Greet the recipient briefly, say you are an AI assistant calling {who}, "
            f"and state the objective.{speak}")


def stt_language(stt_model: str, language: str) -> Optional[str]:
    """The language tag each STT expects for a caller language ('' = auto)."""
    if stt_model == "deepgram/nova-3":
        return "multi" if language in ("", "hi", "en") else language
    return language or None


@dataclass
class CallConfig:
    objective: str = DEFAULT_OBJECTIVE
    recipient_name: str = ""
    clinic_name: str = ""
    greeting: str = ""
    language: str = "hi"
    stt_model: str = STT_MODELS[0]
    reply_model: str = "anthropic/claude-haiku-4.5"
    tts_model: str = DEFAULT_TTS_MODEL
    tts_voice: str = ELEVENLABS_VOICE_ID if ELEVENLABS_READY else ""
    source: str = "voice_lab"
    max_call_seconds: int = MAX_CALL_SECONDS
    # Set by a dispatcher acting for a clinic (Calling Service): books the
    # call's spend to that clinic's D10 ledger. Absent = platform spend.
    usage_context: Optional[Dict[str, str]] = None
    calling_service_callback: Optional[CallCallback] = field(default=None, repr=False)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metadata(cls, raw: str) -> "CallConfig":
        try:
            data = json.loads(raw) if raw else {}
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        cfg = cls()
        # Only permit a dispatcher to shorten the hard cap, never extend it.
        maximum = data.get("max_call_seconds")
        if maximum is not None:
            if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= MAX_CALL_SECONDS:
                raise ValueError("max_call_seconds must be an integer between 1 and 300")
            cfg.max_call_seconds = maximum
        cfg.calling_service_callback = CallCallback.parse(data.get("calling_service_callback"))
        accounting = data.get("accounting_context")
        if isinstance(accounting, dict):
            cfg.extra["accounting"] = {key: value[:512] for key, value in accounting.items()
                if key in ("call_id", "direction", "provider_call_id", "provider_sip_call_id", "recording_policy")
                and isinstance(value, str)}
        for key in ("objective", "recipient_name", "clinic_name", "greeting", "tts_voice", "source"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                setattr(cfg, key, value.strip()[:1000])
        # Unknown or disallowed choices fall back to the default rather than
        # failing the call: a patient is already on the line.
        if data.get("language") in LANGUAGES:
            cfg.language = data["language"]
        if data.get("stt_model") in STT_MODELS:
            cfg.stt_model = data["stt_model"]
        if data.get("reply_model") in REPLY_MODELS:
            cfg.reply_model = data["reply_model"]
        if data.get("tts_model") in TTS_MODELS and (data["tts_model"] != ELEVENLABS_MODEL or ELEVENLABS_READY):
            cfg.tts_model = data["tts_model"]
        if data.get("tts_model") == ELEVENLABS_MODEL and not ELEVENLABS_READY:
            cfg.tts_voice = ""
        # ElevenLabs always speaks with the pinned female voice: a dispatch
        # payload cannot pick another one and cause a gender mismatch.
        if cfg.tts_model == ELEVENLABS_MODEL:
            cfg.tts_voice = ELEVENLABS_VOICE_ID
        if cfg.tts_model != ELEVENLABS_MODEL and not data.get("tts_voice"):
            cfg.tts_voice = ""
        usage = data.get("usage_context")
        if isinstance(usage, dict) and all(
            isinstance(usage.get(k), str) and usage[k].strip() for k in USAGE_CONTEXT_KEYS
        ):
            cfg.usage_context = {k: usage[k].strip()[:512] for k in USAGE_CONTEXT_KEYS}
        if cfg.language == "mr" and cfg.stt_model == "deepgram/nova-3":
            cfg.stt_model = "google/gemini-3.5-transcribe-live"
        if cfg.source == "inbound_receptionist" and (
                not cfg.calling_service_callback or not cfg.usage_context
                or not isinstance(data.get("objective"), str) or not data["objective"].strip()
                or not cfg.greeting):
            raise ValueError("Inbound receptionist dispatch requires objective, greeting, usage context and callback")
        return cfg
