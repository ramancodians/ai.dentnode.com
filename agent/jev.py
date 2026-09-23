"""Small, typed decisions through OpenRouter's Jev Decisions API.

Callers supply trusted tenant scope and meter the returned usage in their own
ledger. This module does not log state, questions, answers, or patient text.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

import httpx

from agent.config import settings


JEV_MODEL = "typesafe/jev-1.13"
_MAX_STATE_BYTES = 32_768
_MAX_CANDIDATES = 20
_MATCH_THRESHOLD = 0.80


class JevError(RuntimeError):
    """A safe failure whose message never contains submitted state or candidates."""


@dataclass(frozen=True)
class DecisionResult:
    model: str
    answers: dict[str, dict[str, Any]]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_ms: int
    provider: str | None
    request_id: str | None


@dataclass(frozen=True)
class TextMatchResult:
    index: int | None
    confidence: float | None
    decision: DecisionResult


def _unit_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _validate_questions(questions: dict[str, Any]) -> None:
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 16:
        raise JevError("Jev questions must contain 1 to 16 typed questions")
    for name, question in questions.items():
        if not isinstance(name, str) or not name or len(name) > 64:
            raise JevError("Jev question name is invalid")
        if not isinstance(question, dict):
            raise JevError("Jev question is invalid")
        kind = question.get("type")
        instructions = question.get("instructions")
        if kind not in ("noul", "choice", "score"):
            raise JevError("Jev question type is unsupported")
        if not isinstance(instructions, str) or not instructions.strip():
            raise JevError("Jev question instructions are required")
        if kind == "choice":
            criteria = question.get("criteria")
            if (
                not isinstance(criteria, dict)
                or not 2 <= len(criteria) <= 21
                or any(
                    not isinstance(key, str)
                    or not key
                    or not isinstance(value, str)
                    or not value.strip()
                    for key, value in criteria.items()
                )
            ):
                raise JevError("Jev choice criteria are invalid")
        elif kind == "score":
            criteria = question.get("criteria")
            if (
                not isinstance(criteria, list)
                or not 2 <= len(criteria) <= 20
                or any(not isinstance(value, str) or not value.strip() for value in criteria)
            ):
                raise JevError("Jev score criteria are invalid")


def _validate_answers(answers: Any, questions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(answers, dict) or answers.keys() != questions.keys():
        raise JevError("Jev response has missing or unexpected answers")
    for name, question in questions.items():
        answer = answers[name]
        kind = question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise JevError("Jev response answer type is invalid")
        if kind == "noul":
            if not _unit_number(answer.get("noul")):
                raise JevError("Jev response noul is invalid")
        elif kind == "choice":
            criteria = question["criteria"]
            if answer.get("choice") not in criteria:
                raise JevError("Jev response choice is outside the criteria")
            if "confidence" in answer and not _unit_number(answer["confidence"]):
                raise JevError("Jev response confidence is invalid")
            if "probabilities" in answer:
                probabilities = answer["probabilities"]
                if (
                    not isinstance(probabilities, dict)
                    or probabilities.keys() != criteria.keys()
                    or not all(_unit_number(value) for value in probabilities.values())
                ):
                    raise JevError("Jev response probabilities are invalid")
        else:
            score = answer.get("score")
            if (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or not 0 <= score <= len(question["criteria"]) - 1
            ):
                raise JevError("Jev response score is invalid")
            if "confidence" in answer and not _unit_number(answer["confidence"]):
                raise JevError("Jev response confidence is invalid")
    return answers


def _decisions_url() -> str:
    base = settings.openrouter_api_base.rstrip("/")
    if not base.endswith("/v1"):
        raise JevError("OpenRouter API base must end in /v1 for Jev")
    return f"{base[:-3]}/alpha/decisions"


async def decide(
    state: str | dict | list,
    questions: dict[str, Any],
    *,
    timeout_secs: float = 5.0,
) -> DecisionResult:
    """Evaluate state with Jev and return validated typed answers and usage."""
    if type(timeout_secs) not in (int, float) or not math.isfinite(timeout_secs) or not 0 < timeout_secs <= 30:
        raise JevError("Jev timeout must be between 0 and 30 seconds")
    if not isinstance(state, (str, dict, list)) or not state:
        raise JevError("Jev state must be a nonempty string, object, or list")
    _validate_questions(questions)
    if not settings.openrouter_api_key:
        raise JevError("OpenRouter API key is not configured")
    body = {"model": JEV_MODEL, "state": state, "questions": questions}
    try:
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise JevError("Jev request is not valid JSON") from exc
    if len(encoded.encode("utf-8")) > _MAX_STATE_BYTES:
        raise JevError("Jev request exceeds the size limit")

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.openrouter_site_url,
        "X-Title": settings.openrouter_app_name,
    }
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout_secs) as client:
            response = await client.post(_decisions_url(), content=encoded, headers=headers)
    except httpx.HTTPError as exc:
        raise JevError("Jev request failed") from exc
    latency_ms = int((time.monotonic() - start) * 1000)
    if response.status_code >= 400:
        raise JevError(f"Jev returned HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError as exc:
        raise JevError("Jev returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise JevError("Jev response is invalid")
    model = data.get("model")
    if not isinstance(model, str) or not model.startswith("typesafe/jev-"):
        raise JevError("Jev response model is invalid")
    answers = _validate_answers(data.get("answers"), questions)
    usage = data.get("usage")
    if not isinstance(usage, dict):
        raise JevError("Jev response usage is invalid")
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        type(input_tokens) is not int
        or input_tokens < 0
        or type(output_tokens) is not int
        or output_tokens < 0
    ):
        raise JevError("Jev response token counts are invalid")
    cost = usage.get("cost")
    if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
        raise JevError("Jev response cost is invalid")
    provider = data.get("provider")
    request_id = data.get("id")
    if provider is not None and not isinstance(provider, str):
        raise JevError("Jev response provider is invalid")
    if request_id is not None and not isinstance(request_id, str):
        raise JevError("Jev response id is invalid")
    return DecisionResult(
        model=model,
        answers=answers,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=float(cost) if cost is not None else None,
        latency_ms=latency_ms,
        provider=provider,
        request_id=request_id,
    )


async def match_text(
    query: str,
    candidates: list[str],
    *,
    timeout_secs: float = 5.0,
) -> TextMatchResult:
    """Choose one candidate or abstain; never silently drop candidates."""
    if not isinstance(query, str) or not query.strip() or len(query) > 256:
        raise JevError("Jev match query is invalid")
    if (
        not isinstance(candidates, list)
        or not 1 <= len(candidates) <= _MAX_CANDIDATES
        or any(not isinstance(value, str) or not value.strip() or len(value) > 256 for value in candidates)
    ):
        raise JevError("Jev match candidates are invalid")
    criteria = {f"candidate_{index}": value for index, value in enumerate(candidates)}
    criteria["none"] = "No single candidate sufficiently matches the query."
    decision = await decide(
        {"query": query},
        {
            "match_exists": {
                "type": "noul",
                "instructions": "Does exactly one listed candidate genuinely refer to the query? Answer no for no match or ambiguity.",
                "criteria": {"false": "No unique match", "true": "One clear match"},
            },
            "match": {
                "type": "choice",
                "instructions": "Choose the one candidate matching the query, or none if absent or ambiguous.",
                "criteria": criteria,
            },
        },
        timeout_secs=timeout_secs,
    )
    choice_answer = decision.answers["match"]
    confidence = choice_answer.get("confidence")
    choice = choice_answer["choice"]
    if (
        decision.answers["match_exists"]["noul"] < _MATCH_THRESHOLD
        or choice == "none"
        or confidence is None
        or confidence < _MATCH_THRESHOLD
    ):
        return TextMatchResult(index=None, confidence=confidence, decision=decision)
    probabilities = choice_answer.get("probabilities")
    if probabilities is not None:
        ranked = sorted(probabilities.values(), reverse=True)
        if len(ranked) > 1 and ranked[0] - ranked[1] < 0.20:
            return TextMatchResult(index=None, confidence=confidence, decision=decision)
    return TextMatchResult(
        index=int(choice.removeprefix("candidate_")),
        confidence=confidence,
        decision=decision,
    )


class JevDecisionAgent:
    """Shared, stateless decision agent used by multiple service agents."""

    decide = staticmethod(decide)
    match_text = staticmethod(match_text)


jev_agent = JevDecisionAgent()
