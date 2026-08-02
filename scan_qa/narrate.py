"""LLM explanation layer — the last stage of the Scan QA pipeline.

The model's ONLY job is to turn findings that geometry already established into
something a technician wants to read. It is given the measurements and told it
may not introduce a defect, change the score, or lower the risk. Everything
quantitative in the output was decided before this module ran.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from agent.config import settings as agent_settings
from agent.openrouter import OpenRouterError, chat_completion

from .pipeline import ScanQAReport

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You write the technician-facing summary for a dental scan QA report.

You are given MEASURED findings from deterministic geometry, plus the case
context. Your job is to explain them clearly, in the order that matters to
someone about to design the case.

Rules — these are absolute:
- Do NOT invent defects. Every claim must trace to a supplied finding.
- Do NOT restate or change the score or risk level; they are already decided.
- Do NOT say a scan is acceptable if a CRITICAL finding is present.
- If the arch scans are clean, SAY SO plainly before discussing problems. A
  report that reads as uniformly alarming gets ignored.
- Distinguish file roles. A buccal bite is a thin registration patch, not an
  arch scan; judge it on whether it can register the arches, nothing else.
- Be specific and short. 3-5 sentences. No markdown, no preamble.

Return only the summary text."""


def _payload(report: ScanQAReport, case_context: Dict[str, Any]) -> str:
    files = []
    for f in report.files:
        g = f.geometry or {}
        files.append(
            {
                "label": f.label,
                "role": f.role,
                "faces": g.get("faces"),
                "shells": g.get("shells"),
                "holes": g.get("hole_count"),
                "watertight": g.get("watertight"),
                "segmentation": (
                    {
                        "teeth_detected": (f.segmentation or {}).get("teeth_detected"),
                        "gingiva_fraction": (f.segmentation or {}).get(
                            "gingiva_fraction"
                        ),
                    }
                    if f.segmentation
                    else None
                ),
                "findings": [
                    {
                        "severity": x.severity,
                        "type": x.type,
                        "title": x.title,
                        "detail": x.detail,
                        "tooth": x.tooth,
                    }
                    for x in f.findings
                ],
            }
        )
    return json.dumps(
        {
            "case_context": case_context,
            "score": round(report.overall_score, 1),
            "risk_level": report.risk_level,
            "files": files,
        },
        indent=2,
        default=str,
    )


async def narrate(
    report: ScanQAReport,
    *,
    case_context: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> ScanQAReport:
    """Attach a readable summary. Never raises — a model outage must not lose
    the measurements, so on failure the report is returned unchanged."""
    try:
        result = await chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _payload(report, case_context or {})},
            ],
            model=model or agent_settings.scan_review_vision_model,
            temperature=0.2,
            max_tokens=600,
        )
        report.summary = result.text.strip()
    except OpenRouterError as exc:
        logger.warning("Scan QA narration failed", extra={"error": str(exc)})
    return report
