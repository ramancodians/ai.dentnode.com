"""Turns ``ai_pricing.py`` into a charge on each D10 usage event.

The charge travels in the event's ``pricing`` block and D10 debits exactly
that amount. Integer credit micros, always rounded up, as D10's own rating is.
"""

import hashlib
import json
import math
from fractions import Fraction
from typing import Any, Dict, Optional

from agent.d10 import ai_pricing as cfg

AI_CALL_FEATURE = "voice_call"
AI_CALL_SESSION_EVENT = "ai.voice_agent_session"
_MICROS = 1_000_000

CONFIG: Dict[str, Any] = {
    "inr_per_credit": cfg.INR_PER_CREDIT,
    "inr_per_usd": cfg.INR_PER_USD,
    "ai_call_inr_per_minute": cfg.AI_CALL_INR_PER_MINUTE,
    "ai_call_round_up_to_minute": cfg.AI_CALL_ROUND_UP_TO_MINUTE,
    "default_multiplier": cfg.DEFAULT_MULTIPLIER,
    "feature_multipliers": dict(cfg.FEATURE_MULTIPLIERS),
}


def _validate(config: Dict[str, Any]) -> None:
    for key in ("inr_per_credit", "inr_per_usd"):
        if not config[key] > 0:
            raise ValueError(f"ai_pricing.py: {key} must be greater than 0")
    if not config["ai_call_inr_per_minute"] >= 0:
        raise ValueError("ai_pricing.py: AI_CALL_INR_PER_MINUTE must be 0 or more")
    multipliers = {"default": config["default_multiplier"], **config["feature_multipliers"]}
    for feature, value in multipliers.items():
        if not value >= 0:
            raise ValueError(f"ai_pricing.py: multiplier for {feature} must be 0 or more")


_validate(CONFIG)
# Identifies this version of the file on every charge (D10's audit rate card).
VERSION = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]


def _exact(value: float) -> Fraction:
    return Fraction(str(value))


def price_event(
    *,
    event_type: str,
    feature: Optional[str],
    cost_micros_usd: Optional[int],
    agent_session_ms: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """The ``pricing`` block for one event, or None when it cannot be priced
    (an AI feature whose provider cost is unknown); D10 then records the event
    unpriced for reconciliation."""
    base = {
        "version": VERSION,
        "inr_per_credit": CONFIG["inr_per_credit"],
        "inr_per_usd": CONFIG["inr_per_usd"],
    }
    if feature == AI_CALL_FEATURE:
        # One per-minute price, carried by the call's session event; its speech,
        # reply-model, voice and phone-line events are included in it.
        if event_type != AI_CALL_SESSION_EVENT:
            return {**base, "rule": "included_in_call_minute", "charged_credits_micros": 0}
        ms = max(0, int(agent_session_ms or 0))
        billed_ms = -(-ms // 60_000) * 60_000 if CONFIG["ai_call_round_up_to_minute"] else ms
        credits = math.ceil(
            Fraction(billed_ms, 60_000) * _exact(CONFIG["ai_call_inr_per_minute"]) * _MICROS
            / _exact(CONFIG["inr_per_credit"])
        )
        return {
            **base,
            "rule": "call_per_minute",
            "charged_credits_micros": credits,
            "inr_per_minute": CONFIG["ai_call_inr_per_minute"],
            "billed_ms": billed_ms,
        }
    if cost_micros_usd is None:
        return None
    multipliers = CONFIG["feature_multipliers"]
    # `in`, not truthiness: a multiplier of 0 (a free feature) is a real setting.
    multiplier = multipliers[feature] if feature in multipliers else CONFIG["default_multiplier"]
    # USD micros × ₹/USD × multiplier ÷ ₹/credit = credit micros.
    credits = math.ceil(
        Fraction(int(cost_micros_usd)) * _exact(CONFIG["inr_per_usd"]) * _exact(multiplier)
        / _exact(CONFIG["inr_per_credit"])
    )
    return {
        **base,
        "rule": "provider_cost_multiplier",
        "charged_credits_micros": credits,
        "multiplier": multiplier,
    }
