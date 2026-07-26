"""Case-from-image extraction — migrated from the Node backend.

Node's controllers/ai/caseFromImage.ts previously called Gemini Vision directly
(`runGeminiExtraction`) to read a photographed dental lab prescription form and
return structured case JSON. That model logic now lives here and runs through
OpenRouter, so it is metered like every other AI call.

Only the prompt, the vision call, and the JSON parse moved. Node still does all
the tenant work — SSRF-checked image fetch, doctor/product matching, confidence
scoring, AiCaseExtraction persistence, and draft entry creation. The system
prompt and the fence-tolerant parser are copied VERBATIM from the Node
implementation, so the structure Node receives is byte-identical in shape.

Side benefit of the move: the direct Gemini path ran on a free-tier key capped at
20 requests/day and returned 500s once exhausted. Routing through OpenRouter
removes that cap.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import settings
from .openrouter import chat_completion

logger = logging.getLogger(__name__)

# Vision payloads are large and the model has to read handwriting; give it more
# headroom than the 60s default used for text-only completions.
_VISION_TIMEOUT_SECS = 120.0

# Copied verbatim from app.dentnode.com/controllers/ai/caseFromImage.ts
# (SYSTEM_PROMPT). Do not reword — extraction behaviour must match.
SYSTEM_PROMPT = """You are an AI that extracts structured dental case details from images of dental lab prescription forms.

Analyze the image carefully. Extract every readable field. This is a dental lab form — doctors write patient details, tooth numbers, product/shade requests, and instructions.

Return a JSON object with this exact structure:

{
  "doctor": {
    "name": "doctor or clinic name found on the form",
    "phone": "phone if visible"
  },
  "patient": {
    "first_name": "first name",
    "last_name": "last name",
    "phone": "phone if visible",
    "gender": "MALE or FEMALE if indicated",
    "age": number or null
  },
  "case": {
    "case_id": "any case/reference number on the form",
    "entry_type": "PHYSICAL or DIGITAL",
    "delivery_date": "date in YYYY-MM-DD if mentioned"
  },
  "work": [
    {
      "product_name": "the product requested e.g. Zirconia Crown, PFM Bridge, Implant, Denture, etc.",
      "teeth": ["11","12","21" etc — FDI notation tooth numbers. If a range like 11-15, expand it. Use FDI: upper right = 11-18, upper left = 21-28, lower right = 41-48, lower left = 31-38. If Palmer notation is used (e.g. UR1, UL3, LL6), convert to FDI."],
      "shade": "shade value like A2, B1, BL3 etc if indicated",
      "shade_guide": "VITA_CLASSIC or VITA_3D if indicated",
      "units": number (default 1),
      "jaw_type": "UPPER or LOWER or BOTH if indicated",
      "instructions": "any special instructions or notes for this work item"
    }
  ],
  "notes": "any general notes or special instructions from the form"
}

Rules:
- Use snake_case exactly as shown
- FDI tooth notation: 11-18 = upper right, 21-28 = upper left, 31-38 = lower left, 41-48 = lower right
- If a tooth is marked/charted on a tooth diagram, include it in the teeth array
- If you see "Rx" or a product description, put it in product_name
- If you cannot read a field, omit it or set it to null
- Return ONLY the JSON object, no markdown, no explanation"""


class CaseExtractionError(RuntimeError):
    """Raised when the model reply cannot be parsed into a case object.

    Mirrors the Node error ("Failed to parse extracted case details from image")
    so the proxy surfaces the same failure to the client.
    """


def parse_extraction_response(raw: str) -> Dict[str, Any]:
    """Port of the Node fence-stripping parse in runGeminiExtraction().

    Node stripped a leading ```/```json fence and a trailing fence, then
    JSON.parse'd. We do the same, then fall back to the outermost {...} slice
    before giving up — a strictly wider net than Node's, never a narrower one.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
            except Exception:
                raise CaseExtractionError(
                    "Failed to parse extracted case details from image"
                )
        else:
            raise CaseExtractionError(
                "Failed to parse extracted case details from image"
            )

    if not isinstance(parsed, dict):
        raise CaseExtractionError(
            "Failed to parse extracted case details from image"
        )
    return parsed


@dataclass
class CaseExtractionResult:
    extracted: Dict[str, Any]
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: Optional[float] = None
    latency_ms: int = 0


async def extract_case_from_image(
    *,
    image_base64: str,
    mime_type: str,
) -> CaseExtractionResult:
    """Run vision extraction on one form image and return the parsed case JSON.

    Node fetches and base64-encodes the image (after an allowlist/SSRF check) and
    passes the bytes through; this service never fetches a URL itself.

    Raises:
        OpenRouterError: model/transport failure.
        CaseExtractionError: reply was not parseable JSON.
    """
    mime = (mime_type or "").strip() or "image/jpeg"
    data_url = f"data:{mime};base64,{image_base64}"

    # Node called Gemini as generateContent([SYSTEM_PROMPT, imagePart]) — both
    # parts in a single user turn, no separate system message. Mirrored here so
    # the model sees the same conversation shape.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    result = await chat_completion(
        messages=messages,
        model=settings.vision_model,
        temperature=0.1,
        timeout_secs=_VISION_TIMEOUT_SECS,
    )

    extracted = parse_extraction_response(result.text)

    return CaseExtractionResult(
        extracted=extracted,
        model=result.model,
        usage=result.usage,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )
