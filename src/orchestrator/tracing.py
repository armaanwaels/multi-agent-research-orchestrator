"""OpenTelemetry tracing, off unless ORCH_TRACING=otel.

Spans go to an OTLP endpoint when OTEL_EXPORTER_OTLP_ENDPOINT is set (Jaeger,
Honeycomb, Langfuse and LangSmith all accept OTLP), otherwise to the console.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

_configured = False


def setup_tracing(mode: str) -> None:
    global _configured
    if _configured or mode != "otel":
        return
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": "research-orchestrator"}))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()
    else:
        exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _configured = True


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    # With no provider configured this is OpenTelemetry's no-op tracer, so it costs nothing.
    with trace.get_tracer("orchestrator").start_as_current_span(name) as s:
        s.set_attributes({k: v for k, v in attributes.items() if v is not None})
        yield s
