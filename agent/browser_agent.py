"""Standalone agent for DNLink test browser control; deliberately not Laby."""
import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from .config import settings
from .tool_schema import slim_tools
from .browser_tools import BROWSER_TOOLS

INSTRUCTION = """You are DentNode Browser Controller, a test-only assistant for open DNLink portal tabs.
You only inspect and interact with pages through the provided browser tools. You are not Laby and must not answer lab, financial, patient, production, reporting, clinical, or business questions.
First call browser_tabs. If no bridge is connected, tell the user to enable AI Browser Control in DNLink settings.
Work one command at a time and obtain its result before the next command. Do not claim a page changed until the result confirms it.
You may only use snapshot, portal-restricted navigation, visible-text clicks, labeled typing, and allowlisted keys. Refuse downloads, uploads, deletion, approval, payment, authentication, credential handling, or any unrepresented action."""

def _model() -> LiteLlm:
    if settings.openrouter_api_key: os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
    return LiteLlm(model=settings.model, api_key=settings.openrouter_api_key or None, api_base=settings.openrouter_api_base)

browser_agent = LlmAgent(name="dnlink_browser", model=_model(), description="Test-only DNLink browser controller.", instruction=INSTRUCTION, tools=slim_tools(BROWSER_TOOLS))