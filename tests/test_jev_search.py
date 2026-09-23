"""Search integration keeps Jev advisory and meters each successful call."""

import json

import pytest

from agent.jev import DecisionResult, TextMatchResult
from agent.jev_search import suggest_search_match, with_search_suggestion


def _suggestion(index=0):
    return TextMatchResult(
        index=index,
        confidence=0.91,
        decision=DecisionResult(
            model="typesafe/jev-1.13", answers={}, input_tokens=80,
            output_tokens=8, cost_usd=0.000003, latency_ms=210,
            provider="TypeSafe", request_id="jev-request-1",
        ),
    )


@pytest.mark.asyncio
async def test_patient_match_uses_only_names_and_preserves_candidates(monkeypatch):
    import agent.jev_search as search

    seen = {}

    async def fake_match(query, labels):
        seen.update(query=query, labels=labels)
        return _suggestion(1)

    monkeypatch.setattr(search.jev_agent, "match_text", fake_match)
    patients = [
        {"patient_id": "private-1", "name": "Asha Rao", "phone": "111"},
        {"patient_id": "private-2", "name": "Asha Roy", "phone": "222"},
    ]
    result = {"patients": patients, "count": 2}
    suggestion = await suggest_search_match("search_patients", "Asha", result)
    annotated = with_search_suggestion(result, suggestion)

    assert seen == {"query": "Asha", "labels": ["Asha Rao", "Asha Roy"]}
    assert annotated["patients"] == patients
    assert annotated["jev_match"] == {
        "row_index": 1, "confidence": 0.91, "advisory_only": True,
    }
    assert "private" not in json.dumps(seen)


@pytest.mark.asyncio
async def test_exact_patient_number_and_case_id_do_not_call_jev(monkeypatch):
    import agent.jev_search as search

    async def unexpected(*_args):
        raise AssertionError("exact lookup should not call Jev")

    monkeypatch.setattr(search.jev_agent, "match_text", unexpected)
    patients = {"patients": [
        {"name": "Asha Rao", "patient_number": "P-42"},
        {"name": "Asha Roy", "patient_number": "P-43"},
    ]}
    cases = {"rows": [
        ["DN-42", "-", "-", "Asha Rao"],
        ["DN-43", "-", "-", "Asha Roy"],
    ]}
    assert await suggest_search_match("search_patients", "P-42", patients) is None
    assert await suggest_search_match("find_case", "DN-42", cases) is None


@pytest.mark.asyncio
async def test_laby_search_records_jev_usage_without_changing_row_ids(monkeypatch):
    import agent.tools as tools

    result = {
        "rows": [["Dr Asha", "First Clinic", "-", "-"],
                 ["Dr Asha", "Second Clinic", "-", "-"]],
        "entity_ids": [{"id": "private-1", "row_index": 0},
                       {"id": "private-2", "row_index": 1}],
    }
    usage = []

    async def fake_tool(*_args, **_kwargs):
        return result

    async def fake_match(*_args):
        return _suggestion(1)

    async def fake_usage(**kwargs):
        usage.append(kwargs)

    monkeypatch.setattr(tools, "_lab_id", lambda _context: "lab-1")
    monkeypatch.setattr(tools, "_user_id", lambda _context: "user-1")
    monkeypatch.setattr(tools, "call_tool", fake_tool)
    monkeypatch.setattr(tools, "suggest_search_match", fake_match)
    monkeypatch.setattr(tools, "report_usage", fake_usage)

    actual = await tools._run(None, "find_doctor", {"query": "Asha"})
    assert actual["entity_ids"] == result["entity_ids"]
    assert actual["jev_match"]["row_index"] == 1
    assert usage[0]["model"] == "typesafe/jev-1.13"
    assert usage[0]["cost"] == 0.000003
    assert usage[0]["lab_id"] == "lab-1"


@pytest.mark.asyncio
async def test_d10_search_records_jev_usage_before_returning_results(monkeypatch):
    from agent.d10 import runner
    from agent.d10.context import D10RequestContext
    from agent.openrouter import ChatResult

    class Outbox:
        def __init__(self):
            self.events = []

        async def enqueue(self, event):
            self.events.append(event)

    context = D10RequestContext(
        clinic_id="clinic-1", user_id="user-1", conversation_id="conv-1",
        actor_id="staff-1", actor_role="DENTIST", timezone="Asia/Calcutta",
        source_message_id="message-1", correlation_id="corr-1",
        causation_id="cause-1", reservation_id="reserve-1",
    )
    model_responses = iter([
        ChatResult(
            text="", model="deepseek/model", usage={}, latency_ms=20,
            message={"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-1", "type": "function", "function": {
                    "name": "search_patients", "arguments": json.dumps({"query": "Asha"}),
                },
            }]},
        ),
        ChatResult(
            text="Which Asha?", model="deepseek/model", usage={},
            latency_ms=20, message={"role": "assistant", "content": "Which Asha?"},
        ),
    ])
    candidates = {"patients": [{"patient_id": "p1", "name": "Asha Rao"},
                               {"patient_id": "p2", "name": "Asha Roy"}]}

    async def fake_catalog(_context):
        return []

    async def fake_chat(**_kwargs):
        return next(model_responses)

    async def fake_tool(**_kwargs):
        return candidates

    async def fake_match(*_args):
        return _suggestion(0)

    monkeypatch.setattr(runner, "fetch_tool_catalog", fake_catalog)
    monkeypatch.setattr(runner, "chat_completion", fake_chat)
    monkeypatch.setattr(runner, "call_tool", fake_tool)
    monkeypatch.setattr(runner, "suggest_search_match", fake_match)
    outbox = Outbox()
    events = [event async for event in runner.run_d10_turn(
        message="Find Asha", context=context, history=None, outbox=outbox,
    )]

    assert [event["model_call_index"] for event in outbox.events] == [1, 2, 3]
    assert outbox.events[1]["model"] == "typesafe/jev-1.13"
    assert outbox.events[1]["provider_cost_micros_usd"] == 3
    assert events[-1]["message"] == "Which Asha?"
    tool_result = next(event for event in events if event["type"] == "tool_result")
    assert tool_result["result"]["patients"] == candidates["patients"]
    assert tool_result["result"]["jev_match"]["advisory_only"] is True
