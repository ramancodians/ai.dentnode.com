"""RX (prescription) review feature — migrated from the Node backend.

Node's controllers/ai/rxReview.ts previously called Gemini directly with three
sub-agent prompts (completeness, clinical, instructions) to audit a dental work
prescription. That model logic now lives here and runs through OpenRouter (the
service's single LLM gateway), so it is metered like every other AI call.

Only the prompt construction, the model call, and the parse-into-findings step
moved. The Node side still fetches the entry from the DB, builds the per-agent
user input (buildCaseInput), combines/scores the findings, and persists the
AiReview record. The three system prompts and the response-parsing logic are
copied VERBATIM from the Node implementation — behaviour is identical, only the
provider/plumbing changed.

Gemini used `responseMimeType: application/json` to force JSON output. OpenRouter
models are steered the same way by the prompts themselves ("Return ONLY valid
JSON — no markdown, no text outside the JSON") plus the identical, fence-tolerant
parser below, so the parsed structure Node expects is unchanged.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .openrouter import ChatResult, OpenRouterError, chat_completion

# ─── Sub-agent system prompts ─────────────────────────────────────────────────
# Copied verbatim from app.dentnode.com/controllers/ai/rxReview.ts. Do not reword.

COMPLETENESS_PROMPT = """You are DentAI, a dental lab prescription completeness auditor.
You receive a dental work prescription and must check every work item for completeness.
Return ONLY valid JSON — no markdown, no text outside the JSON.

Schema:
{
  "findings": [
    {
      "id": "c1",
      "category": "COMPLETENESS",
      "severity": "INFO" | "WARNING" | "CRITICAL",
      "title": "<short title>",
      "detail": "<what is missing or incomplete and on which work item>",
      "recommendation": "<what the prescribing dentist or technician should add>"
    }
  ]
}

Rules:
- CRITICAL: Missing shade for any restoration (crown, veneer, bridge, inlay); missing tooth notation for multi-unit work; zero or missing unit count
- WARNING: Shade written without number (e.g. just "A" instead of "A2"); jaw type not specified for full-arch work; no patient name on complex cases
- INFO: Missing secondary shade, no stump shade noted for high translucency restorations
- If all items are complete, return { "findings": [] }"""

CLINICAL_PROMPT = """You are DentAI, a dental lab clinical compatibility specialist.
You receive a dental work prescription and must flag clinical or material compatibility issues.
Return ONLY valid JSON — no markdown, no text outside the JSON.

Schema:
{
  "findings": [
    {
      "id": "cl1",
      "category": "CLINICAL" | "MATERIAL",
      "severity": "INFO" | "WARNING" | "CRITICAL",
      "title": "<short title>",
      "detail": "<description of the compatibility issue>",
      "recommendation": "<corrective action>"
    }
  ]
}

Rules:
- CRITICAL: Unit count exceeds anatomically possible (>14 for full arch); implant prosthetic ordered without implant system noted; impossible tooth-product combination
- WARNING: PFM ordered for anterior zone without note; unit count unusually high for the product; shade incompatible with product category (e.g. very dark shade for full-zirconia)
- MATERIAL: Material choice unusual for the clinical zone (posterior vs anterior), shade guide not standard
- INFO: A better alternative material for the specified use case
- If no issues found, return { "findings": [] }"""

INSTRUCTIONS_PROMPT = """You are DentAI, a dental lab instruction quality reviewer.
You receive lab instructions from a dental work prescription and assess if they are clear, complete, and actionable for a lab technician.
Return ONLY valid JSON — no markdown, no text outside the JSON.

Schema:
{
  "findings": [
    {
      "id": "i1",
      "category": "PRESCRIPTION",
      "severity": "INFO" | "WARNING" | "CRITICAL",
      "title": "<short title>",
      "detail": "<what is unclear, missing, or contradictory>",
      "recommendation": "<how to improve the instruction>"
    }
  ]
}

Rules:
- CRITICAL: No instructions at all for a complex case (implant bridge, full-arch, >4 units); contradictory instructions
- WARNING: Vague instructions ("make it natural looking", "match existing teeth" with no reference); missing occlusion notes for bite-sensitive cases; missing contact requirements
- INFO: Suggestions that would help production (requested surface texture, stain details, try-in protocol preference)
- If instructions are clear and sufficient, return { "findings": [] }"""


# ─── Parsing ──────────────────────────────────────────────────────────────────
# Verbatim port of parseFindingsResponse from rxReview.ts. Same fence-stripping
# and same {"findings": []} fallback, so the parsed shape Node receives is identical.


def parse_findings_response(raw: str) -> Dict[str, Any]:
    cleaned = raw
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    try:
        import json

        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                import json

                return json.loads(cleaned[start : end + 1])
            except Exception:
                pass
        return {"findings": []}


# ─── Sub-agent runner ─────────────────────────────────────────────────────────


async def _run_sub_agent(
    system_prompt: str, user_input: str
) -> Tuple[Dict[str, Any], Optional[ChatResult]]:
    """Run one sub-agent completion. Mirrors runSubAgent() in rxReview.ts.

    On any model/transport failure it degrades gracefully to {"findings": []}
    (exactly as the Node try/catch did), so one sub-agent failing never fails the
    whole review. Returns (parsed, chat_result | None) so the caller can meter.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    try:
        result = await chat_completion(
            messages=messages,
            model=settings.model,
            temperature=0.1,
        )
        return parse_findings_response(result.text), result
    except OpenRouterError:
        return {"findings": []}, None


@dataclass
class RxReviewResult:
    completeness: Dict[str, Any]
    clinical: Dict[str, Any]
    instructions: Dict[str, Any]
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: Optional[float] = None
    latency_ms: int = 0
    any_ok: bool = False


async def generate_rx_review(
    *,
    completeness: str,
    clinical: str,
    instructions: str,
) -> RxReviewResult:
    """Run the three prescription sub-agents and return their parsed findings.

    The three per-agent user inputs are built by Node (buildCaseInput) and passed
    through verbatim. All three run in parallel, each degrading to {"findings": []}
    on failure — identical to the Node implementation. Token usage and OpenRouter
    cost are aggregated across the successful calls for a single metering event.
    """
    comp_res, clin_res, inst_res = await asyncio.gather(
        _run_sub_agent(COMPLETENESS_PROMPT, completeness),
        _run_sub_agent(CLINICAL_PROMPT, clinical),
        _run_sub_agent(INSTRUCTIONS_PROMPT, instructions),
    )

    chat_results: List[ChatResult] = [
        r for (_, r) in (comp_res, clin_res, inst_res) if r is not None
    ]

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost_usd: Optional[float] = None
    latency_ms = 0
    for r in chat_results:
        usage["prompt_tokens"] += int(r.usage.get("prompt_tokens", 0) or 0)
        usage["completion_tokens"] += int(r.usage.get("completion_tokens", 0) or 0)
        usage["total_tokens"] += int(r.usage.get("total_tokens", 0) or 0)
        if r.cost_usd is not None:
            cost_usd = (cost_usd or 0.0) + r.cost_usd
        latency_ms = max(latency_ms, r.latency_ms)

    return RxReviewResult(
        completeness=comp_res[0],
        clinical=clin_res[0],
        instructions=inst_res[0],
        model=settings.model,
        usage=usage,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        any_ok=len(chat_results) > 0,
    )
