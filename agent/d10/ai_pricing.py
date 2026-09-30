"""What D10 clinics are charged for AI. Edit this file to change prices.

All prices are in rupees; clinics pay in D10 credits, so every price is
converted with ``INR_PER_CREDIT``. A change applies to usage recorded after the
deploy that ships it; nothing already recorded is re-priced. Every charge
carries this file's version hash and the rule that produced it, and D10 keeps
each version as an audit rate card, so a past bill can always be explained.

Deliberately a reviewed file, not env vars.
"""

# Rupee value of one D10 credit. Packs sell at ₹2.00 / ₹1.80 / ₹1.50 per credit.
INR_PER_CREDIT = 1.8

# Converts provider costs (OpenRouter and LiveKit bill in USD) to rupees.
INR_PER_USD = 88

# AI phone calls: one all-inclusive price per connected minute, covering the AI
# (speech, reply model, voice) and the phone line. The call's individual
# components are then not charged again.
AI_CALL_INR_PER_MINUTE = 5
# True: every started minute is a full minute. False: per second.
AI_CALL_ROUND_UP_TO_MINUTE = True

# Every other AI feature: provider cost (what DentNode paid OpenRouter) times a
# multiplier. 3 = charge three times our cost.
DEFAULT_MULTIPLIER = 3
# Per-feature overrides, keyed by feature name as shown in the AI cost report.
FEATURE_MULTIPLIERS: dict[str, float] = {
    # "assistant": 2,
    # "notes_transcription": 3,
}
