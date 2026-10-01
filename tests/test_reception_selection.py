"""Offline decision-boundary checks; no model, database or provider calls."""
import unittest
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.reception.selection import select_tools
from agent.jev import JevError
from fastapi import HTTPException

gateway = importlib.import_module("agent.reception.router")


def declarations(*names):
    return [{"type": "function", "function": {"name": name}} for name in names]


class SelectionTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return gateway.Selection(scope={"platform": "app", "context": {
            "lab_id": "lab", "user_id": "admin", "correlation_id": "trace"}}, query="Book a visit")

    async def test_auth_rejects_before_catalog_or_model(self):
        with patch.object(gateway.settings, "internal_key", "expected"), patch.object(gateway, "catalog", AsyncMock()) as catalog:
            with self.assertRaises(HTTPException) as error:
                await gateway.select_reception_tools(self.request(), "incorrect", None)
        self.assertEqual(error.exception.status_code, 401)
        catalog.assert_not_called()

    async def test_failure_returns_full_catalog_and_meters_attempt(self):
        tools = declarations("find_case", "call_summary")
        with patch.object(gateway.settings, "internal_key", "expected"), \
             patch.object(gateway, "catalog", AsyncMock(return_value=tools)), \
             patch.object(gateway, "select_tools", AsyncMock(side_effect=JevError("unavailable"))), \
             patch.object(gateway, "report_usage", AsyncMock()) as meter, \
             patch.object(gateway, "execute", AsyncMock()) as execute:
            result = await gateway.select_reception_tools(self.request(), "expected", None)
        self.assertEqual(result["tools"], tools)
        self.assertEqual(result["selection"]["outcome"], "unavailable")
        self.assertEqual(meter.call_args.kwargs["status"], "error")
        execute.assert_not_called()

    async def test_shortlist_keeps_prerequisites_and_unknown_capabilities(self):
        tools = declarations("book_appointment", "find_appointment_slots", "call_summary",
                             "search_patients", "handoff_to_human", "future_tool")
        with patch("agent.reception.selection.jev_agent.match_text", AsyncMock(return_value=SimpleNamespace(index=0))) as model:
            selected, outcome, _ = await select_tools("Book an appointment", tools)
        self.assertEqual(outcome, "suggested")
        self.assertEqual([t["function"]["name"] for t in selected],
                         ["book_appointment", "find_appointment_slots", "search_patients", "handoff_to_human", "future_tool"])
        self.assertNotIn("future_tool", str(model.call_args))

    async def test_ambiguous_intent_preserves_entire_catalog(self):
        tools = declarations("book_appointment", "call_summary")
        with patch("agent.reception.selection.jev_agent.match_text", AsyncMock(return_value=SimpleNamespace(index=None))):
            selected, outcome, _ = await select_tools("Booking and previous calls", tools)
        self.assertIs(selected, tools)
        self.assertEqual(outcome, "unclear")

    async def test_single_family_skips_model(self):
        tools = declarations("book_appointment", "find_appointment_slots")
        with patch("agent.reception.selection.jev_agent.match_text", AsyncMock()) as model:
            selected, outcome, _ = await select_tools("Booking", tools)
        model.assert_not_called()
        self.assertIs(selected, tools)
        self.assertEqual(outcome, "not_needed")

    async def test_invalid_candidate_cannot_remove_tools(self):
        tools = declarations("book_appointment", "call_summary")
        with patch("agent.reception.selection.jev_agent.match_text", AsyncMock(return_value=SimpleNamespace(index=99))):
            selected, outcome, _ = await select_tools("Booking", tools)
        self.assertIs(selected, tools)
        self.assertEqual(outcome, "unclear")
