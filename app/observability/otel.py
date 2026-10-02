from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from functools import wraps

# Optional OpenTelemetry - graceful degradation if not installed
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import SpanKind, Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    SpanKind = None
    Status = None
    StatusCode = None

from app.settings import Settings

# Global tracer
_tracer = None
_initialized = False


def init_tracing(settings: Settings) -> None:
    """Initialize OpenTelemetry if available and configured."""
    global _tracer, _initialized
    if _initialized or not OTEL_AVAILABLE:
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    )
    if not endpoint:
        return  # No endpoint configured, skip

    resource = Resource.create({"service.name": "tutorcopilot-agent"})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("tutorcopilot")
    _initialized = True


def get_tracer():
    return _tracer


@contextlib.contextmanager
def span(name: str, kind: SpanKind | None = None, attributes: dict | None = None):
    """Context manager for a traced span."""
    if not OTEL_AVAILABLE or _tracer is None:
        yield None
        return

    span_obj = _tracer.start_span(name, kind=kind or SpanKind.INTERNAL)
    if attributes:
        for k, v in attributes.items():
            span_obj.set_attribute(k, v)
    try:
        yield span_obj
    except Exception as e:
        span_obj.set_status(Status(StatusCode.ERROR, str(e)))
        span_obj.record_exception(e)
        raise
    finally:
        span_obj.end()


def traced(name: str | None = None, kind: SpanKind | None = None, attributes: dict | None = None):
    """Decorator to trace a function."""

    def decorator(func: Callable):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            with span(span_name, kind, attributes):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            with span(span_name, kind, attributes):
                return await func(*args, **kwargs)

        import asyncio

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Convenience traced operations
def llm_span(name):
    return traced(name, SpanKind.CLIENT) if OTEL_AVAILABLE else _noop(name)


def tool_span(name):
    return traced(name, SpanKind.INTERNAL) if OTEL_AVAILABLE else _noop(name)


def db_span(name):
    return traced(name, SpanKind.CLIENT) if OTEL_AVAILABLE else _noop(name)


def _noop(name):
    return lambda f: f
