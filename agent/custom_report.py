"""Custom accounting report builder — prompt → report spec.

Node (app.dentnode.com services/customReports) sends the lab user's request
and the catalog of datasets/fields it can run. The model answers with a report
SPEC only: dataset, columns, filters, date preset, grouping, aggregates, sort
and an optional chart in Laby's { kind, x, y } shape. It never sees lab data
and never writes SQL; Node validates the spec against its allowlist and runs
it itself. On a failed validation Node calls again once with the errors.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import settings
from .openrouter import chat_completion

_TIMEOUT_SECS = 60.0

SYSTEM_PROMPT = """You turn a dental lab's request for an accounting report into a JSON report spec.

You only choose from the catalog you are given: datasets, their field keys, filter ops,
date presets, aggregate functions and chart kinds. Never invent a field key. Never write SQL.

Spec shape:
{
  "version": 1,
  "dataset": "<dataset key>",
  "columns": ["<field key>", ...],              // row-level report; leave [] when grouping
  "filters": [{"field": "<key>", "op": "<op>", "value": <value>}],
  "dateRange": {"preset": "<preset>"} or {"preset": "custom", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "groupBy": ["<field key>", ...],              // up to 3; makes a summary report
  "aggregates": [{"fn": "sum|avg|min|max|count", "field": "<key>"}],  // count takes no field
  "sort": {"key": "<output column key>", "direction": "asc|desc"},
  "limit": <number, optional, for "top N">,
  "chart": {"kind": "bar|line|pie", "x": "<output column key>", "y": "<output column key>"} or null
}

Rules:
- Output column keys are the "columns" for a row report, or the groupBy keys followed by
  aggregate keys for a summary. An aggregate's key is "<fn>_<field>" (e.g. "sum_billed"), or "count".
- sort.key, chart.x and chart.y must be output column keys. chart.y must be numeric.
- Filter ops by field type: string: eq, neq, in, not_in, contains, is_empty, not_empty;
  enum: eq, neq, in, not_in (values from the field's options); number/currency: eq, neq, gt, gte,
  lt, lte, between ([from, to]); date: gt, gte, lt, lte, between (ISO dates); boolean: eq (true/false).
  in/not_in take a list.
- dateRange applies to the dataset's dateField. The "clients" dataset has none: it is balances as of now.
- Default dateRange is {"preset": "this_month"} unless the user names a period or wants all history.
  Financial year (this_fy/last_fy) runs April to March. last_6_months/last_12_months are whole
  calendar months including the current one. Prefer a preset; for custom use plain YYYY-MM-DD dates
  (inclusive, whole days). Use "today" from the request for relative dates.
- "Collected"/"received" money excludes write-offs: on payments filter kind = "Collection".
  Write-offs are kind = "Write-off" (method DISCOUNT).
- What a client owes now is the "balance" field of the "clients" dataset.
- Use a time bucket field (…Day, …Month, …Quarter, …Year) to group over time.
- Add a chart when the user asks for one, or when the report is a summary that reads better as one:
  line for trends over time buckets, bar to compare categories, pie for a share of a total
  with few categories. Row-level lists get "chart": null.
- For "top N" use sort desc on the amount and limit N.
- If a current spec is given, the request is a change to it: keep everything the user did not ask to change.

Reply with ONLY a JSON object, no prose and no code fences:
{"title": "<short report name>", "summary": "<one sentence describing the report>", "spec": { ... }}"""


class CustomReportParseError(RuntimeError):
    """The model reply was not a usable JSON object."""


@dataclass
class CustomReportResult:
    result: Dict[str, Any]
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: Optional[float] = None
    latency_ms: int = 0


def build_user_prompt(
    *,
    prompt: str,
    catalog: Dict[str, Any],
    today: str,
    current_spec: Optional[Dict[str, Any]] = None,
    previous_spec: Optional[Any] = None,
    errors: Optional[List[str]] = None,
) -> str:
    parts = [
        f"Today: {today}",
        "Catalog (JSON):",
        json.dumps(catalog, ensure_ascii=False),
    ]
    if current_spec:
        parts += ["Current spec (the user wants to change it):", json.dumps(current_spec, ensure_ascii=False)]
    parts += ["Request:", prompt.strip()]
    if errors:
        parts += [
            "Your previous spec was rejected:",
            json.dumps(previous_spec, ensure_ascii=False),
            "Errors:",
            "\n".join(f"- {e}" for e in errors),
            "Return a corrected reply.",
        ]
    return "\n".join(parts)


def parse_reply(raw: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise CustomReportParseError("Model reply contained no JSON object")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except Exception as exc:
            raise CustomReportParseError(f"Could not parse report JSON: {exc}")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("spec"), dict):
        raise CustomReportParseError("Model reply had no spec object")
    return parsed


async def generate_custom_report_spec(
    *,
    prompt: str,
    catalog: Dict[str, Any],
    today: str,
    current_spec: Optional[Dict[str, Any]] = None,
    previous_spec: Optional[Any] = None,
    errors: Optional[List[str]] = None,
) -> CustomReportResult:
    """Raises OpenRouterError on model failure, CustomReportParseError on a bad reply."""
    result = await chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    prompt=prompt,
                    catalog=catalog,
                    today=today,
                    current_spec=current_spec,
                    previous_spec=previous_spec,
                    errors=errors,
                ),
            },
        ],
        model=settings.model,
        temperature=0.1,
        timeout_secs=_TIMEOUT_SECS,
    )
    return CustomReportResult(
        result=parse_reply(result.text),
        model=result.model,
        usage=result.usage,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )
