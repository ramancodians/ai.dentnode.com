"""Contract checks for Jev's typed Decisions API client."""

import pytest
import respx
from httpx import Response

from agent.config import settings
from agent.jev import JevError, decide, match_text


def _response(answers):
    return {
        "id": "gen-dec-123",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": answers,
        "usage": {"input_tokens": 50, "output_tokens": 8, "cost": 0.00002},
    }


@pytest.mark.asyncio
@respx.mock
async def test_decide_sends_typed_questions_and_preserves_usage(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_base", "https://openrouter.test/api/v1")
    route = respx.post("https://openrouter.test/api/alpha/decisions").mock(
        return_value=Response(200, json=_response({"ok": {"type": "noul", "noul": 0.96}}))
    )
    result = await decide(
        {"patient": "private text"},
        {"ok": {"type": "noul", "instructions": "Is this the right record?"}},
    )

    assert route.called
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer sk-or-v1-test"
    assert request.headers["HTTP-Referer"] == settings.openrouter_site_url
    assert request.content
    assert b'"model": "typesafe/jev-1.13"' in request.content
    assert result.answers["ok"]["noul"] == 0.96
    assert result.model == "typesafe/jev-1.13-20260917"
    assert result.provider == "TypeSafe"
    assert result.request_id == "gen-dec-123"
    assert result.input_tokens == 50
    assert result.output_tokens == 8
    assert result.cost_usd == 0.00002
    assert result.latency_ms >= 0


@pytest.mark.asyncio
@respx.mock
async def test_decide_rejects_malformed_answer_without_echoing_patient(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_base", "https://openrouter.test/api/v1")
    respx.post("https://openrouter.test/api/alpha/decisions").mock(
        return_value=Response(
            200,
            json=_response({"match": {"type": "choice", "choice": "unknown"}}),
        )
    )
    with pytest.raises(JevError) as error:
        await decide(
            "Private Patient Name",
            {
                "match": {
                    "type": "choice",
                    "instructions": "Choose a record",
                    "criteria": {"one": "Person One", "none": "No person"},
                }
            },
        )
    assert "Private Patient Name" not in str(error.value)
    assert "Person One" not in str(error.value)


@pytest.mark.asyncio
@respx.mock
async def test_decide_raises_safe_http_error(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_base", "https://openrouter.test/api/v1")
    respx.post("https://openrouter.test/api/alpha/decisions").mock(
        return_value=Response(400, text="private patient details")
    )
    with pytest.raises(JevError, match="HTTP 400") as error:
        await decide("Private Patient Name", {"ok": {"type": "noul", "instructions": "Match?"}})
    assert "Private Patient Name" not in str(error.value)
    assert "private patient details" not in str(error.value)


@pytest.mark.asyncio
@respx.mock
async def test_match_text_requires_absolute_and_unambiguous_match(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_base", "https://openrouter.test/api/v1")
    route = respx.post("https://openrouter.test/api/alpha/decisions").mock(
        return_value=Response(
            200,
            json=_response(
                {
                    "match_exists": {"type": "noul", "noul": 0.98},
                    "match": {
                        "type": "choice",
                        "choice": "candidate_1",
                        "confidence": 0.95,
                        "probabilities": {"candidate_0": 0.03, "candidate_1": 0.95, "none": 0.02},
                    },
                }
            ),
        )
    )
    result = await match_text("Raman", ["Ram", "Raman"])
    assert result.index == 1
    assert result.confidence == 0.95
    assert result.decision.cost_usd == 0.00002
    assert b'"none": "No single candidate' in route.calls.last.request.content


@pytest.mark.asyncio
@respx.mock
async def test_match_text_abstains_when_absolute_match_is_weak(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_base", "https://openrouter.test/api/v1")
    respx.post("https://openrouter.test/api/alpha/decisions").mock(
        return_value=Response(
            200,
            json=_response(
                {
                    "match_exists": {"type": "noul", "noul": 0.51},
                    "match": {
                        "type": "choice",
                        "choice": "candidate_0",
                        "confidence": 0.95,
                        "probabilities": {"candidate_0": 0.95, "none": 0.05},
                    },
                }
            ),
        )
    )
    assert (await match_text("Raman", ["Ram"])).index is None


@pytest.mark.asyncio
@respx.mock
async def test_match_text_abstains_for_close_runners_up(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_base", "https://openrouter.test/api/v1")
    respx.post("https://openrouter.test/api/alpha/decisions").mock(
        return_value=Response(
            200,
            json=_response(
                {
                    "match_exists": {"type": "noul", "noul": 0.99},
                    "match": {
                        "type": "choice",
                        "choice": "candidate_0",
                        "confidence": 0.90,
                        "probabilities": {"candidate_0": 0.52, "candidate_1": 0.47, "none": 0.01},
                    },
                }
            ),
        )
    )
    assert (await match_text("Raman", ["Ram", "Raman"])).index is None


@pytest.mark.asyncio
async def test_match_text_rejects_more_than_twenty_candidates():
    with pytest.raises(JevError, match="candidates"):
        await match_text("Raman", ["name"] * 21)
