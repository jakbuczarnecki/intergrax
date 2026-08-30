# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenTelemetry spans for RAG retrieve and ingest hot paths (M-RAG.27)."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Iterator, Mapping, Optional

from intergrax.contracts.instrumentation_span_attributes import (
    InstrumentationSpan,
    SpanAttributeValue,
    merge_safe_span_attributes,
    normalize_span_attribute_value,
)

RAG_OTEL_TRACER_NAME = "intergrax.rag"

RAG_OTEL_SPAN_NAMES: tuple[str, ...] = (
    "rag.retrieve",
    "rag.retrieve.single_pass",
    "rag.ingest",
    "rag.ingest.load",
    "rag.ingest.chunk",
    "rag.ingest.index",
    "rag.ingest.graph_index",
)

_rag_otel_spans_enabled_override: Optional[bool] = None


def _env_rag_otel_spans_enabled() -> bool:
    return os.getenv("INTERGRAX_RAG_OTEL_SPANS_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def set_rag_otel_spans_enabled(enabled: bool | None) -> None:
    global _rag_otel_spans_enabled_override
    _rag_otel_spans_enabled_override = enabled


def is_rag_otel_spans_enabled() -> bool:
    if _rag_otel_spans_enabled_override is not None:
        return _rag_otel_spans_enabled_override
    return _env_rag_otel_spans_enabled()


@contextmanager
def rag_span(
    name: str,
    *,
    attributes: Optional[Mapping[str, SpanAttributeValue | None]] = None,
) -> Iterator[None]:
    """
    Emit an OpenTelemetry span when RAG OTel spans are enabled.

    Unlike aggregated RAG metrics (``INTERGRAX_RAG_METRICS_ENABLED``), spans are
    on the default observability spine and enabled unless explicitly disabled.
    """
    if not is_rag_otel_spans_enabled():
        yield
        return

    span_cm: AbstractContextManager[InstrumentationSpan | None] = nullcontext()
    span_attributes: dict[str, SpanAttributeValue] = {}
    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        tracer = trace.get_tracer(RAG_OTEL_TRACER_NAME)
        span_attributes = merge_safe_span_attributes(caller_attributes=attributes)
        span_cm = tracer.start_as_current_span(name)
    except ImportError:
        yield
        return
    except Exception:
        span_cm = nullcontext()

    with span_cm as span:
        if span is not None and span_attributes:
            try:
                for key, value in span_attributes.items():
                    normalized = normalize_span_attribute_value(value)
                    if normalized is not None:
                        span.set_attribute(key, normalized)
            except Exception:
                pass
        try:
            yield
        except Exception as exc:
            if span is not None:
                try:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                except Exception:
                    pass
            raise
