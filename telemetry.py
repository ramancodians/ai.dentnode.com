"""Privacy-preserving OpenTelemetry tracing for the production AI service.

Only operational HTTP metadata is exported: method, templated inbound route,
status, protocol, and outbound server name/port. Bodies, headers, query
strings, concrete path parameters, client addresses, exception messages,
events, baggage, and arbitrary attributes are removed at the exporter boundary.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.textmap import Getter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

_INSTRUMENTATION_SCOPE = "dentnode.ai.http"
_configured = False


class _TraceHeaderGetter(Getter[Mapping[str, str]]):
    def get(self, carrier: Mapping[str, str], key: str) -> list[str] | None:
        value = carrier.get(key.lower())
        return [value] if value else None

    def keys(self, carrier: Mapping[str, str]) -> list[str]:
        return list(carrier.keys())


_trace_header_getter = _TraceHeaderGetter()
_trace_context = TraceContextTextMapPropagator()


def _safe_method(attributes: Mapping[str, Any]) -> str:
    value = attributes.get("http.request.method") or attributes.get("http.method")
    if not isinstance(value, str):
        return "REQUEST"
    method = value.upper()
    return method if method.isascii() and method.isalpha() and len(method) <= 16 else "REQUEST"


def _safe_route(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith("/") or len(value) > 200:
        return None
    if "?" in value or "#" in value or any(ord(char) < 32 for char in value):
        return None
    # Only registered route templates are supplied by our middleware. A route
    # containing a non-template segment could be a concrete patient identifier.
    segments = value.split("/")
    if any(segment and not segment.startswith("{") and not segment.replace("-", "").replace("_", "").isalnum() for segment in segments):
        return None
    return value


def _safe_server_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    hostname = value.rstrip(".").lower()
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789.-")
    if not hostname or len(hostname) > 253 or any(char not in allowed for char in hostname):
        return None
    return hostname


def _set_safe_httpx_destination(span: trace.Span, request_info: Any) -> None:
    """Copy only the destination origin from HTTPX's request object."""

    server_name = _safe_server_name(getattr(request_info.url, "host", None))
    if server_name:
        span.set_attribute("server.address", server_name)
    port = getattr(request_info.url, "port", None)
    if isinstance(port, int) and 1 <= port <= 65535:
        span.set_attribute("server.port", port)


def _httpx_request_hook(span: trace.Span, request_info: Any) -> None:
    _set_safe_httpx_destination(span, request_info)


async def _async_httpx_request_hook(span: trace.Span, request_info: Any) -> None:
    _set_safe_httpx_destination(span, request_info)


def sanitize_span(span: ReadableSpan) -> ReadableSpan:
    """Return an export-only copy containing allowlisted operational metadata."""

    source = dict(span.attributes or {})
    attributes: dict[str, Any] = {}
    method = _safe_method(source)
    attributes["http.request.method"] = method

    status_code = source.get("http.response.status_code", source.get("http.status_code"))
    if isinstance(status_code, int) and 100 <= status_code <= 599:
        attributes["http.response.status_code"] = status_code

    protocol = source.get("network.protocol.version", source.get("http.flavor"))
    if isinstance(protocol, str) and protocol in {"1.0", "1.1", "2", "3"}:
        attributes["network.protocol.version"] = protocol

    if span.kind is SpanKind.SERVER and span.instrumentation_scope and span.instrumentation_scope.name == _INSTRUMENTATION_SCOPE:
        route = _safe_route(source.get("http.route"))
        if route:
            attributes["http.route"] = route
    else:
        route = None

    if span.kind is SpanKind.CLIENT:
        server_name = _safe_server_name(source.get("server.address") or source.get("net.peer.name"))
        if server_name:
            attributes["server.address"] = server_name
        server_port = source.get("server.port", source.get("net.peer.port"))
        if isinstance(server_port, int) and 1 <= server_port <= 65535:
            attributes["server.port"] = server_port
    else:
        server_name = None

    if span.kind is SpanKind.SERVER:
        name = f"HTTP {method}{' ' + route if route else ''}"
    elif span.kind is SpanKind.CLIENT:
        name = f"HTTP {method}{' ' + server_name if server_name else ''}"
    else:
        name = "operation"

    # Status descriptions and events commonly contain exception messages or
    # concrete URLs. Keep the status code, timing, trace relationship, and no
    # free-form data. Links are also omitted because their attributes are open.
    clean_status = Status(span.status.status_code)
    return ReadableSpan(
        name=name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=(),
        links=(),
        kind=span.kind,
        instrumentation_scope=span.instrumentation_scope,
        start_time=span.start_time,
        end_time=span.end_time,
        status=clean_status,
    )


class PrivacyFilteringSpanExporter(SpanExporter):
    """Enforce the privacy boundary immediately before OTLP serialization."""

    def __init__(self, delegate: SpanExporter) -> None:
        self._delegate = delegate

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._delegate.export(tuple(sanitize_span(span) for span in spans))

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._delegate.force_flush(timeout_millis)


class SafeHttpTracingMiddleware:
    """Trace FastAPI requests without reading request or response content."""

    def __init__(self, app: Any, tracer: trace.Tracer) -> None:
        self.app = app
        self.tracer = tracer

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "REQUEST")).upper()
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
            # traceparent carries only trace/span ids and sampling. tracestate
            # and baggage are intentionally not accepted because their values
            # are free-form and could carry patient or tenant data.
            if key.lower() == b"traceparent"
        }
        parent_context: Context = _trace_context.extract(headers, getter=_trace_header_getter)
        attributes: dict[str, Any] = {"http.request.method": method}
        protocol = scope.get("http_version")
        if protocol in {"1.0", "1.1", "2", "3"}:
            attributes["network.protocol.version"] = protocol

        status_code: int | None = None

        async def traced_send(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                candidate = message.get("status")
                if isinstance(candidate, int):
                    status_code = candidate
            await send(message)

        with self.tracer.start_as_current_span(
            f"HTTP {_safe_method(attributes)}",
            context=parent_context,
            kind=SpanKind.SERVER,
            attributes=attributes,
        ) as span:
            try:
                await self.app(scope, receive, traced_send)
            except BaseException:
                span.set_status(Status(StatusCode.ERROR))
                raise
            finally:
                if status_code is not None:
                    span.set_attribute("http.response.status_code", status_code)
                    if status_code >= 500:
                        span.set_status(Status(StatusCode.ERROR))
                route = getattr(scope.get("route"), "path", None)
                safe_route = _safe_route(route)
                if safe_route:
                    span.set_attribute("http.route", safe_route)
                    span.update_name(f"HTTP {_safe_method(attributes)} {safe_route}")


def _validated_endpoint() -> str | None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    if not endpoint:
        return None
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT must be a plain HTTP(S) URL without credentials or query data")
    return endpoint


def _sample_ratio() -> float:
    raw = os.getenv("OTEL_TRACES_SAMPLER_ARG", "0.10")
    try:
        ratio = float(raw)
    except ValueError as exc:
        raise RuntimeError("OTEL_TRACES_SAMPLER_ARG must be a number from 0 to 1") from exc
    if not 0.0 <= ratio <= 1.0:
        raise RuntimeError("OTEL_TRACES_SAMPLER_ARG must be a number from 0 to 1")
    return ratio


def configure_telemetry(app: FastAPI) -> bool:
    """Configure one process-wide provider; return False when OTLP is unset."""

    global _configured
    if _configured:
        return True

    endpoint = _validated_endpoint()
    if endpoint is None or os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true":
        logger.info("OpenTelemetry tracing disabled (no production OTLP endpoint configured)")
        return False

    # Import here so local/test startup without an endpoint stays side-effect free.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    service_name = os.getenv("OTEL_SERVICE_NAME", "ai.dentnode.com").strip()
    if not service_name or len(service_name) > 100:
        raise RuntimeError("OTEL_SERVICE_NAME must be 1-100 characters")
    service_version = os.getenv("OTEL_SERVICE_VERSION", "unknown").strip() or "unknown"

    resource = Resource(
        {
            "service.name": service_name,
            "service.namespace": "dentnode",
            "service.version": service_version[:128],
            "deployment.environment.name": "production",
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(_sample_ratio())),
    )
    exporter = PrivacyFilteringSpanExporter(OTLPSpanExporter(endpoint=endpoint))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    set_global_textmap(_trace_context)

    tracer = provider.get_tracer(_INSTRUMENTATION_SCOPE)
    app.add_middleware(SafeHttpTracingMiddleware, tracer=tracer)
    HTTPXClientInstrumentor().instrument(
        tracer_provider=provider,
        request_hook=_httpx_request_hook,
        async_request_hook=_async_httpx_request_hook,
    )
    _configured = True
    logger.info("OpenTelemetry tracing enabled", extra={"service": service_name})
    return True
