"""Rejected-cases report — migrated from the Node backend cron.

Node's cron-jobs/rejectedCasesReport.ts previously called Gemini directly with a
text summary of every rejected case plus up to 12 attached screenshots, and asked
for a structured operations report. That model logic now lives here and runs
through OpenRouter, so the (per-lab, vision-heavy, therefore expensive) call is
metered like every other AI call.

Node still owns everything else: selecting rejected entries, building the case
snapshots, fetching the screenshots, and persisting RejectedCasesReport.

Both prompts are copied VERBATIM from the Node implementation.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import settings
from .openrouter import chat_completion

logger = logging.getLogger(__name__)

# Matches the Node cron's window; used only for prompt text.
WINDOW_DAYS = 30

# Up to 12 images plus a long case list — this is the heaviest single call the
# service makes.
_TIMEOUT_SECS = 240.0


class RejectedCasesParseError(RuntimeError):
    """Raised when the model reply is not usable JSON.

    Node's cron did a bare JSON.parse and let the failure mark the report FAILED,
    so a parse failure must stay a hard error here too.
    """


# Copied verbatim from cron-jobs/rejectedCasesReport.ts (buildSystemPrompt).
SYSTEM_PROMPT = """You are a dental lab operations analyst. You receive data about cases that were rejected during intake review at a dental laboratory.

Analyze the rejection patterns and return ONLY valid JSON — no markdown, no prose outside the JSON object.

Schema:
{
  "summary": "<2-3 sentence executive overview of the rejection situation this period>",
  "health_score": <0-100, where 100 means zero rejections and 0 means catastrophic rejection rate>,
  "trend": "IMPROVING" | "STABLE" | "DECLINING",
  "patterns": [
    {
      "name": "<short pattern name>",
      "count": <number of cases matching this pattern>,
      "percentage": <% of total rejections>,
      "description": "<what is happening>",
      "root_cause": "<the underlying reason why this keeps happening>",
      "fault_distribution": { "lab": <0-100>, "client": <0-100> }
    }
  ],
  "action_items": [
    {
      "priority": "HIGH" | "MEDIUM" | "LOW",
      "title": "<concise action title>",
      "description": "<exactly what to do, step by step if needed>",
      "expected_impact": "<what measurable outcome will improve>",
      "assigned_to": "lab" | "client" | "both"
    }
  ],
  "case_analyses": [
    {
      "entry_id": "<id>",
      "case_id": "<display id or null>",
      "doctor_name": "<name>",
      "work_type": "<comma-separated product names>",
      "rejection_date": "<ISO date>",
      "rejection_reason": "<concise summary of why it was rejected>",
      "fault_owner": "lab" | "client" | "unknown",
      "image_observations": "<what was visually observed in attached images, or null if no images>",
      "suggestion": "<specific fix for this case or to prevent recurrence>"
    }
  ]
}

Guidelines:
- health_score: 100 = 0 rejections; 80 = rare; 60 = occasional; 40 = frequent; below 40 = systemic problem
- Prioritize patterns by frequency. Surface the single most impactful action_item first.
- For image_observations: describe specific visible defects (scan artefacts, missing anatomy, occlusion issues, etc.)
- action_items should be concrete, not vague ("Train staff on X scanner bite registration" not "Improve quality")
- fault_distribution must sum to 100"""


def build_user_prompt(cases: List[Dict[str, Any]], lab_name: str) -> str:
    """Verbatim port of buildUserPrompt() in rejectedCasesReport.ts."""
    lines: List[str] = [
        f"Lab: {lab_name}",
        f"Analysis period: Last {WINDOW_DAYS} days (ending today)",
        f"Total rejected cases: {len(cases)}",
        "",
        "=== REJECTED CASES ===",
    ]

    for i, c in enumerate(cases):
        work_types = c.get("work_types") or []
        image_urls = c.get("image_urls") or []
        lines.append(f"\n[Case {i + 1}]")
        lines.append(f"  Entry ID: {c.get('entry_id')}")
        lines.append(f"  Display ID: {c.get('case_id') or 'N/A'}")
        lines.append(f"  Doctor/Clinic: {c.get('doctor_name')}")
        lines.append(
            f"  Work type(s): {', '.join(work_types) if work_types else 'Not specified'}"
        )
        lines.append(f"  Rejected on: {c.get('rejected_at')}")
        lines.append(f"  Rejection message: {c.get('rejection_message') or '(no message)'}")
        lines.append(f"  Fault attributed to: {c.get('fault_owner') or 'unspecified'}")
        lines.append(f"  Screenshots attached: {len(image_urls)}")

    return "\n".join(lines)


def parse_report(raw: str) -> Dict[str, Any]:
    """Strip fences and parse. Raises RejectedCasesParseError if unusable."""
    cleaned = raw
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise RejectedCasesParseError("Model reply contained no JSON object")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except Exception as exc:
            raise RejectedCasesParseError(f"Could not parse report JSON: {exc}")

    if not isinstance(parsed, dict):
        raise RejectedCasesParseError("Model reply was not a JSON object")
    return parsed


@dataclass
class RejectedCasesResult:
    report: Dict[str, Any]
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: Optional[float] = None
    latency_ms: int = 0


async def generate_rejected_cases_report(
    *,
    cases: List[Dict[str, Any]],
    lab_name: str,
    images: Optional[List[Dict[str, Any]]] = None,
) -> RejectedCasesResult:
    """Analyse this lab's rejected cases and return the structured report.

    Args:
        cases:   Case snapshots built by Node.
        lab_name: Lab display name, used in the prompt header.
        images:  Optional [{image_base64, mime_type}] screenshots. Node caps the
                 count (12 at time of writing) to bound vision cost.

    Raises:
        OpenRouterError: model/transport failure.
        RejectedCasesParseError: reply was not usable JSON.
    """
    images = images or []

    # Text first, then images — OpenRouter's recommended ordering.
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": build_user_prompt(cases, lab_name)}
    ]
    for img in images:
        mime = (img.get("mime_type") or "").strip() or "image/jpeg"
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{img['image_base64']}"
                },
            }
        )

    result = await chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        # Vision model: the report reasons over rejection screenshots.
        model=settings.vision_model,
        temperature=0.3,
        timeout_secs=_TIMEOUT_SECS,
    )

    return RejectedCasesResult(
        report=parse_report(result.text),
        model=result.model,
        usage=result.usage,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )
