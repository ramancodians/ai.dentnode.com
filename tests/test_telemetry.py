from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import SpanKind, Status, StatusCode

from telemetry import (
    PrivacyFilteringSpanExporter,
    SafeHttpTracingMiddleware,
    _INSTRUMENTATION_SCOPE,
    _httpx_request_hook,
    _validated_endpoint,
)


def _provider_and_exporter():
    delegate = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource({"service.name": "telemetry-test"}),
        sampler=ALWAYS_ON,
    )
    provider.add_span_processor(SimpleSpanProcessor(PrivacyFilteringSpanExporter(delegate)))
    return provider, delegate


def test_fastapi_trace_uses_route_template_and_never_patient_data():
    provider, delegate = _provider_and_exporter()
    app = FastAPI()
    app.add_middleware(
        SafeHttpTracingMiddleware,
        tracer=provider.get_tracer(_INSTRUMENTATION_SCOPE),
    )

    @app.get("/patients/{patient_id}")
    def patient(patient_id: str):
        return {"patient_id": patient_id}

    secret = "patient-72-secret"
    with TestClient(app) as client:
        response = client.get(
            f"/patients/{secret}?diagnosis=private",
            headers={
                "authorization": "Bearer secret-token",
                "x-forwarded-for": "203.0.113.42",
                "tracestate": f"vendor={secret}",
                "baggage": f"patient={secret}",
            },
        )

    assert response.status_code == 200
    spans = [span for span in delegate.get_finished_spans() if span.kind is SpanKind.SERVER]
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "HTTP GET /patients/{patient_id}"
    assert dict(span.attributes) == {
        "http.request.method": "GET",
        "http.response.status_code": 200,
        "network.protocol.version": "1.1",
        "http.route": "/patients/{patient_id}",
    }
    serialized = span.to_json()
    for forbidden in (secret, "diagnosis", "private", "authorization", "secret-token", "203.0.113.42"):
        assert forbidden not in serialized


def test_export_boundary_drops_free_form_attributes_events_and_status_description():
    provider, delegate = _provider_and_exporter()
    tracer = provider.get_tracer("third.party.instrumentation")
    secret = "patient-and-token-secret"

    with tracer.start_as_current_span(
        f"GET /patients/{secret}",
        kind=SpanKind.CLIENT,
        attributes={
            "http.method": "POST",
            "http.url": f"https://api.example.com/patients/{secret}?token={secret}",
            "http.status_code": 503,
            "net.peer.name": "api.example.com",
            "net.peer.port": 443,
            "http.request.header.authorization": f"Bearer {secret}",
            "client.address": "203.0.113.9",
        },
    ) as span:
        span.add_event("exception", {"exception.message": secret})
        span.set_status(Status(StatusCode.ERROR, secret))

    exported = delegate.get_finished_spans()
    assert len(exported) == 1
    clean = exported[0]
    assert clean.name == "HTTP POST api.example.com"
    assert dict(clean.attributes) == {
        "http.request.method": "POST",
        "http.response.status_code": 503,
        "server.address": "api.example.com",
        "server.port": 443,
    }
    assert clean.events == ()
    assert clean.links == ()
    assert clean.status.status_code is StatusCode.ERROR
    assert clean.status.description is None
    assert secret not in clean.to_json()
    assert "203.0.113.9" not in clean.to_json()


def test_httpx_instrumentation_exports_timing_without_url_or_query():
    provider, delegate = _provider_and_exporter()
    instrumentor = HTTPXClientInstrumentor()
    secret = "patient-outbound-secret"

    transport = httpx.MockTransport(lambda _request: httpx.Response(201, json={"ok": True}))
    with httpx.Client(transport=transport) as client:
        instrumentor.instrument_client(
            client,
            tracer_provider=provider,
            request_hook=_httpx_request_hook,
        )
        try:
            response = client.post(
                f"https://api.example.com/patients/{secret}?token={secret}",
                headers={"authorization": f"Bearer {secret}"},
                json={"patient": secret},
            )
        finally:
            instrumentor.uninstrument_client(client)

    assert response.status_code == 201
    spans = [span for span in delegate.get_finished_spans() if span.kind is SpanKind.CLIENT]
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "HTTP POST api.example.com"
    assert span.attributes["http.request.method"] == "POST"
    assert span.attributes["http.response.status_code"] == 201
    assert span.attributes["server.address"] == "api.example.com"
    assert span.start_time is not None
    assert span.end_time is not None
    assert span.start_time <= span.end_time
    serialized = span.to_json()
    assert secret not in serialized
    assert "authorization" not in serialized
    assert "token=" not in serialized


def test_otlp_endpoint_rejects_credentials_and_query_data(monkeypatch):
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "http://user:password@collector:4318/v1/traces?token=secret",
    )
    try:
        _validated_endpoint()
    except RuntimeError as exc:
        assert "without credentials or query data" in str(exc)
    else:
        raise AssertionError("unsafe OTLP endpoint was accepted")
