# © Artur Czarnecki. All rights reserved.

"""OTel span names for context engineering (CE-9.2 / CE-MAINT-01)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional

CE_OTEL_TRACER_NAME = "intergrax.context"

CE_OTEL_SPAN_NAMES: tuple[str, ...] = (
    "context.engine.assemble",
    "context.provider.collect",
    "context.budget.allocate",
)

_ce_otel_spans_enabled_override: Optional[bool] = None


def _env_ce_otel_spans_enabled() -> bool:
    return os.getenv("INTERGRAX_CE_OTEL_SPANS_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def set_ce_otel_spans_enabled(enabled: bool | None) -> None:
    global _ce_otel_spans_enabled_override
    _ce_otel_spans_enabled_override = enabled


def is_ce_otel_spans_enabled() -> bool:
    if _ce_otel_spans_enabled_override is not None:
        return _ce_otel_spans_enabled_override
    return _env_ce_otel_spans_enabled()


def _normalize_attribute_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


@contextmanager
def context_span(
    name: str,
    *,
    attributes: Optional[Mapping[str, Any]] = None,
) -> Iterator[None]:
    """Emit an OpenTelemetry span when CE OTel spans are enabled."""
    if name not in CE_OTEL_SPAN_NAMES:
        raise ValueError(f"unknown context span: {name}")
    if not is_ce_otel_spans_enabled():
        yield
        return

    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        yield
        return

    tracer = trace.get_tracer(CE_OTEL_TRACER_NAME)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                normalized = _normalize_attribute_value(value)
                if normalized is not None:
                    span.set_attribute(key, normalized)
        try:
            yield
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise
