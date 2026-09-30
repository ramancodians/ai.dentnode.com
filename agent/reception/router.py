"""Stateless platform tool gateway. Domain rules stay in the owning backend."""
import hmac
import time
from uuid import uuid4
from typing import Annotated, Any, Literal

import httpx
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from agent.config import settings
from agent.d10.context import D10RequestContext
from agent.d10.client import fetch_tool_catalog, D10ToolError
from agent.jev import JEV_MODEL, JevError
from agent.usage import report_usage
from agent.d10.usage_outbox import UsageOutbox, build_model_usage_event
from .selection import select_tools

router = APIRouter(prefix="/reception/tools", tags=["reception-tools"])


class AppContext(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    lab_id: str = Field(min_length=1, max_length=256)
    user_id: str = Field(min_length=1, max_length=256)
    correlation_id: str = Field(min_length=1, max_length=256)


class D10Scope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    platform: Literal["d10"]
    context: D10RequestContext


class AppScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    platform: Literal["app"]
    context: AppContext


Scope = Annotated[D10Scope | AppScope, Field(discriminator="platform")]


class Execution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: Scope
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    parameters: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str = Field(min_length=1, max_length=256)


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: Scope
    query: str = Field(min_length=1, max_length=256)


def authenticate(scope: D10Scope | AppScope, app_key: str | None, d10_key: str | None):
    expected, provided = ((settings.d10_internal_key, d10_key)
                          if scope.platform == "d10" else (settings.internal_key, app_key))
    if not expected or not provided or not hmac.compare_digest(expected, provided):
        raise HTTPException(401, "Invalid platform internal key")


async def backend_request(scope: D10Scope | AppScope, suffix: str, payload: dict):
    """Never retry execution: a lost response may follow a successful mutation."""
    if scope.platform == "d10":
        base = settings.d10_internal_base_url.rstrip("/") + "/internal/agent/tools"
        headers = {"x-d10-internal-key": settings.d10_internal_key}
    else:
        base = settings.node_base_url.rstrip("/") + "/internal/laby-tools/reception"
        headers = {"x-internal-key": settings.internal_key}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(base + suffix, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Backend unavailable; execution outcome may be unknown. Reconcile before retrying.") from exc
    if response.is_error:
        # Do not forward backend stacks, SQL, or credentials into model context.
        status = response.status_code if response.status_code in (400, 401, 403, 404, 409, 422, 429) else 502
        raise HTTPException(status, "Backend rejected the tool request" if status != 502 else "Backend tool failed; reconcile before retrying")
    try:
        body = response.json()
    except ValueError as exc:
        raise HTTPException(502, "Invalid backend response") from exc
    if not isinstance(body, dict) or body.get("success") is False:
        raise HTTPException(502, "Backend tool failed")
    return body


async def catalog(scope: D10Scope | AppScope):
    if scope.platform == "d10":
        try:
            return await fetch_tool_catalog(scope.context)
        except D10ToolError as exc:
            raise HTTPException(502, "D10 catalog unavailable or context rejected") from exc
    body = await backend_request(scope, "/catalog", {"context": scope.context.model_dump()})
    return body["tools"]


@router.post("/catalog")
async def tool_catalog(scope: Scope, x_internal_key: str | None = Header(default=None),
                       x_d10_internal_key: str | None = Header(default=None)):
    authenticate(scope, x_internal_key, x_d10_internal_key)
    return {"success": True, "version": "1", "platform": scope.platform, "tools": await catalog(scope)}


async def execute(request: Execution):
    """In-process entrypoint; only trusted orchestration supplies request.scope."""
    reserved = {"clinic_id", "clinicId", "lab_id", "labId", "user_id", "userId",
                "actor_id", "actorId", "actor_role", "actorRole", "conversation_id",
                "conversationId", "context", "correlation_id", "correlationId",
                "source_message_id", "sourceMessageId", "reservation_id", "reservationId"}
    if reserved.intersection(request.parameters):
        raise HTTPException(422, "Identity must come from trusted scope, not tool parameters")
    declarations = await catalog(request.scope)
    allowed = {tool["function"]["name"] for tool in declarations}
    if request.name not in allowed:
        raise HTTPException(404, "Tool unavailable for this context")
    payload = {"context": request.scope.context.model_dump(exclude_none=True),
               "parameters": request.parameters, "tool_call_id": request.tool_call_id}
    body = await backend_request(request.scope, "/" + request.name, payload)
    return {"success": True, "platform": request.scope.platform, "tool": request.name,
            "tool_call_id": request.tool_call_id, "result": body.get("result", body)}


@router.post("/execute")
async def execute_tool(request: Execution, x_internal_key: str | None = Header(default=None),
                       x_d10_internal_key: str | None = Header(default=None)):
    authenticate(request.scope, x_internal_key, x_d10_internal_key)
    return await execute(request)


@router.post("/select")
async def select_reception_tools(request: Selection, x_internal_key: str | None = Header(default=None),
                                 x_d10_internal_key: str | None = Header(default=None)):
    authenticate(request.scope, x_internal_key, x_d10_internal_key)
    tools = await catalog(request.scope)
    context = request.scope.context
    if request.scope.platform == "d10":
        # D10 catalog role filtering is only a discovery hint. Validate persisted
        # scope via a read-only tool before permitting a billable decision.
        await backend_request(request.scope, "/get_reception_status", {
            "context": context.model_dump(exclude_none=True), "parameters": {},
            "tool_call_id": "reception-scope-check"})
    started = time.monotonic()
    decision = None
    status = "ok"
    try:
        shortlist, outcome, match = await select_tools(request.query, tools)
        decision = match.decision if match else None
    except JevError:
        shortlist, outcome, status = tools, "unavailable", "error"
    latency = decision.latency_ms if decision else int((time.monotonic() - started) * 1000)
    if decision or status == "error":
        usage = {"prompt_tokens": decision.input_tokens, "completion_tokens": decision.output_tokens} if decision else None
        cost = decision.cost_usd if decision else None
        model = decision.model if decision else JEV_MODEL
        request_id = decision.request_id if decision else None
        if request.scope.platform == "d10":
            outbox = UsageOutbox(settings.d10_usage_outbox_path)
            await outbox.initialize()
            # Each attempt is separately billable, even on a repeated selection.
            metering_context = context.model_copy(update={"correlation_id": str(uuid4()), "causation_id": context.correlation_id})
            await outbox.enqueue(build_model_usage_event(
                context=metering_context, model_call_index=1, attempt=1,
                provider=(decision.provider or "TypeSafe") if decision else "openrouter",
                model=model, provider_request_id=request_id, usage=usage, raw_usage=None,
                latency_ms=latency, status=status, cost_usd=cost,
                feature="assistant_decision", error_code="JevError" if status == "error" else None))
        else:
            await report_usage(feature="reception_tool_selection", lab_id=context.lab_id,
                user_id=context.user_id, model=model, usage=usage, cost=cost,
                cost_source="openrouter" if cost is not None else "estimated",
                latency_ms=latency, status=status, request_id=request_id,
                meta={"outcome": outcome, "catalog_count": len(tools), "selected_count": len(shortlist)})
    return {"success": True, "platform": request.scope.platform, "tools": shortlist,
            "selection": {"outcome": outcome, "advisory_only": True,
                          "catalog_count": len(tools), "selected_count": len(shortlist),
                          "latency_ms": latency, "cost_usd": decision.cost_usd if decision else None},
            "fallback": "Use the full catalog and the existing conversation agent when clarification or other tools are needed."}
