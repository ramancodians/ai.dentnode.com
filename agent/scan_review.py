"""Oral Scanner AI Review (3D scan quality) — migrated from the Node backend.

Node's controllers/ai/dentalReview.ts previously ran this two ways depending on
the configured provider:
  - Gemini, natively multi-image (one image part per arch view), or
  - NVIDIA Llama 3.2 Vision, which accepts at most ONE image per prompt, so all
    views were composited into a single labeled grid by controllers/ai/montage.ts.

Both paths now collapse into one: OpenRouter accepts multiple images in a single
message (separate content entries), so every view is sent as its own image and
the montage workaround is retired. That removes the grid-legend prompt hack and
gives the model full-resolution views instead of shrunken tiles.

Node still renders the previews — STL/mesh work is geometry, not AI, and stays in
the Node backend (stlPreviews.ts). This service receives ready-made PNGs.

The system prompt is copied VERBATIM from the Node implementation, and the
response parser is a faithful port, so the structure Node persists is unchanged.

Ordering note: OpenRouter recommends text parts BEFORE image parts. The old
Gemini path interleaved a "View: <name>" label before each image; we instead put
one ordered legend in the leading text block and then send the images in exactly
that order, which preserves the labelling without fighting the parser.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import settings
from .openrouter import chat_completion

logger = logging.getLogger(__name__)

# A dozen arch renders is a big payload and a slow read; allow well over the
# 60s text default.
_SCAN_TIMEOUT_SECS = 180.0

# Copied verbatim from app.dentnode.com/controllers/ai/dentalReview.ts
# (SYSTEM_PROMPT). Do not reword — review behaviour must match.
SYSTEM_PROMPT = """You are the Oral Scanner AI Review, a dental laboratory 3D scan quality reviewer.
You receive rendered screenshots of dental arch scans taken at up to 6 standard angles per arch:
Buccal, Right, Left, Lingual, Upper (occlusal), Lower (occlusal).
You also receive case context: product type, shade, notes, and expected delivery.
Return ONLY valid JSON — no markdown, no prose outside the JSON object.

Schema:
{
  "summary": "<2-3 sentence overview of scan quality and key issues>",
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "overall_score": <0-100>,
  "findings": [
    {
      "id": "f1",
      "category": "OCCLUSION" | "MESH_QUALITY" | "CLEARANCE" | "COMPLETENESS" | "PRINT_READINESS" | "DIMENSIONS",
      "severity": "INFO" | "WARNING" | "CRITICAL",
      "title": "<short title>",
      "detail": "<what you observe in the images or case data>",
      "recommendation": "<what the lab should do>"
    }
  ],
  "flags": ["<short_tags>"]
}

Visual assessment guidelines:
- COMPLETENESS: Does the scan cover all tooth surfaces? Missing teeth, large voids, or cropped arches are warnings.
- MESH_QUALITY: Rough/noisy surfaces, floating fragments, or obvious scan artifacts are problems.
- OCCLUSION: From the buccal view, do upper and lower arches appear in a plausible bite relationship?
- CLEARANCE: Very minimal space between arches (near-contact) or excessively open bites should be flagged.
- PRINT_READINESS: Obvious holes, open surfaces, or non-manifold geometry visible in renders.
- DIMENSIONS: A scan that appears unusually small or large relative to a normal arch warrants a note.

Scoring: 90–100 = excellent; 70–89 = minor issues; 50–69 = notable problems; below 50 = critical.
If no scan images are available, review based on case context only and note the limitation."""


def parse_ai_response(raw: str) -> Optional[Dict[str, Any]]:
    """Port of parseAiResponse() in dentalReview.ts.

    Returns None (not an exception) when nothing parseable is found — Node's
    caller already degrades to raw text in that case, and a scan review is an
    optional add-on to the case review, so a soft failure is correct here.
    """
    cleaned = raw
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def build_content(
    *,
    case_context: Dict[str, Any],
    views: List[Dict[str, Any]],
    preview_errors: List[str],
    today: str,
) -> List[Dict[str, Any]]:
    """Build the multimodal user content: all text first, then every image.

    Mirrors buildMultimodalContent() in dentalReview.ts, with the per-image
    labels hoisted into a single ordered legend (see module docstring).
    """
    lines = [
        f"Today: {today}",
        "",
        f"Case context:\n{json.dumps(case_context, indent=2, ensure_ascii=False, default=str)}",
    ]

    if preview_errors:
        lines.append("")
        lines.append("Warnings during preview generation:\n" + "\n".join(preview_errors))

    if not views:
        lines.append("")
        lines.append(
            "No scan previews could be generated. Please review based on case "
            "context only."
        )
        return [{"type": "text", "text": "\n".join(lines)}]

    legend = "\n".join(
        f"{i + 1}. {v.get('label') or 'view'}" for i, v in enumerate(views)
    )
    lines.append("")
    lines.append(
        f"{len(views)} scan render(s) follow, in this exact order:\n{legend}\n\n"
        'Each is one standard view of a dental arch scan, labeled "<ARCH> · <view>". '
        "Assess scan quality across every image."
    )

    content: List[Dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    for v in views:
        mime = (v.get("mime_type") or "").strip() or "image/png"
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{v['image_base64']}"},
            }
        )
    return content


@dataclass
class ScanReviewResult:
    summary: Optional[str]
    risk_level: Optional[str]
    overall_score: Optional[float]
    findings: List[Any]
    flags: List[Any]
    raw_response: str
    model: str
    parsed: bool
    usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: Optional[float] = None
    latency_ms: int = 0


async def generate_scan_review(
    *,
    case_context: Dict[str, Any],
    views: List[Dict[str, Any]],
    preview_errors: Optional[List[str]] = None,
    today: str,
) -> ScanReviewResult:
    """Run the vision scan review over every provided arch render.

    Raises OpenRouterError on model/transport failure so the endpoint can meter
    status="error" and return a clean 502.
    """
    content = build_content(
        case_context=case_context,
        views=views,
        preview_errors=preview_errors or [],
        today=today,
    )

    result = await chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        model=settings.scan_review_vision_model,
        temperature=0.2,
        max_tokens=2048,
        timeout_secs=_SCAN_TIMEOUT_SECS,
    )

    parsed = parse_ai_response(result.text)
    if parsed is None:
        logger.warning(
            "Scan review output was not parseable JSON",
            extra={"chars": len(result.text)},
        )

    findings = (parsed or {}).get("findings")
    flags = (parsed or {}).get("flags")

    return ScanReviewResult(
        summary=(parsed or {}).get("summary"),
        risk_level=(parsed or {}).get("risk_level"),
        overall_score=(parsed or {}).get("overall_score"),
        findings=findings if isinstance(findings, list) else [],
        flags=flags if isinstance(flags, list) else [],
        raw_response=result.text,
        model=result.model,
        parsed=parsed is not None,
        usage=result.usage,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )
