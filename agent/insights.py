"""Clinic metrics "insights" feature — migrated from the Node backend.

Node's controllers/ai/index.ts previously called Gemini directly to turn a
clinic's structured performance metrics into a short prose report. That logic
now lives here and runs through OpenRouter (the service's single LLM gateway),
so it is metered like every other AI call. The prompt and the user-content
shape are copied VERBATIM from the Node implementation — behaviour is identical,
only the provider/plumbing changed.
"""

import json
from typing import Any, Dict, Optional

from .config import settings
from .openrouter import ChatResult, chat_completion

# Copied verbatim from app.dentnode.com/controllers/ai/index.ts
# (DENTAL_LAB_SYSTEM_PROMPT). Do not reword — public behaviour must match.
DENTAL_LAB_SYSTEM_PROMPT = """You are an expert dental laboratory business analyst.
You receive structured performance data for a dental clinic and produce concise,
actionable insights. Focus on trends, anomalies, revenue health, delivery performance,
and top product patterns. Write in clear professional prose with short bullet points
where helpful. Do not fabricate numbers — only reference what is in the data."""


def build_user_content(data: Dict[str, Any], question: Optional[str]) -> str:
    """Build the same user message the Node code built.

    Node:
        const userContent = [
          question ? `Question / focus area: ${question}\\n`
                   : "Please generate a comprehensive performance report.\\n",
          "Clinic metrics data (JSON):",
          JSON.stringify(data, null, 2),
        ].join("\\n");
    """
    header = (
        f"Question / focus area: {question}\n"
        if question
        else "Please generate a comprehensive performance report.\n"
    )
    return "\n".join(
        [
            header,
            "Clinic metrics data (JSON):",
            json.dumps(data, indent=2, ensure_ascii=False),
        ]
    )


async def generate_insights(
    *,
    data: Dict[str, Any],
    question: Optional[str] = None,
) -> ChatResult:
    """Run the insights completion through OpenRouter and return the result.

    Uses the service's configured default text model (settings.model). Raises
    OpenRouterError on model failure so the endpoint can meter status="error"
    and return a clean error.
    """
    messages = [
        {"role": "system", "content": DENTAL_LAB_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(data, question)},
    ]
    return await chat_completion(messages=messages, model=settings.model)
