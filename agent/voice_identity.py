"""The one voice DentNode calls speak with, and the persona that must match it.

Raman, 2026-10-30: every ElevenLabs line uses Shivi V. ElevenLabs labels it
gender=female, language=hi, young, conversational (verified via
GET /v1/voices/BYoWSRudHCnfmjFrtcSW). It is pinned here in code, not read from
env or call metadata, so no stale secret or dispatch payload can switch the
voice. Hindi verbs are gendered, so every spoken prompt must also make the
assistant speak as a woman; a female voice saying "कर सकता हूँ" is the mistake
this module exists to prevent.
"""

VOICE_ID = "BYoWSRudHCnfmjFrtcSW"
VOICE_NAME = "Shivi V"
VOICE_GENDER = "female"

FEMININE_SPEECH_RULE = (
    "You speak in a woman's voice: always use feminine first-person Hindi forms "
    "(मैं कर सकती हूँ, मैं बता रही हूँ, मैं भेज दूँगी — never सकता/रहा/दूँगा about yourself)."
)
