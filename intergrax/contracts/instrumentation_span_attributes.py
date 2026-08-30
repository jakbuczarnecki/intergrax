# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Safe correlation attributes for optional OTel instrumentation spans (HARDEN-3E)."""

from __future__ import annotations

from typing import Mapping, Protocol, TypeAlias

from intergrax.contracts.execution_identity import (
    peek_active_execution_id,
    peek_active_execution_identity,
)

INTERGRAX_RUN_ID_ATTR = "intergrax.run_id"
INTERGRAX_ATTEMPT_ID_ATTR = "intergrax.attempt_id"
INTERGRAX_EXECUTION_ID_ATTR = "intergrax.execution_id"

SpanAttributeScalar: TypeAlias = str | bool | int | float
SpanAttributeValue: TypeAlias = SpanAttributeScalar


class InstrumentationSpan(Protocol):
    """Minimal OTel span surface used by optional instrumentation helpers."""

    def set_attribute(self, key: str, value: SpanAttributeScalar) -> None: ...

    def set_status(self, status: object) -> None: ...

    def record_exception(self, exception: BaseException) -> None: ...

_UNSAFE_ATTRIBUTE_KEY_FRAGMENTS: tuple[str, ...] = (
    "content",
    "prompt",
    "body",
    "query_text",
    "document_text",
    "chunk_text",
    "secret",
    "password",
    "credential",
    "api_key",
)


def active_execution_span_attributes() -> dict[str, str]:
    """Return canonical execution identity attributes for derived tracing spans.

  Never mints synthetic IDs; only reads the active execution identity context.
    """
    attributes: dict[str, str] = {}
    bound = peek_active_execution_identity()
    if bound is not None:
        run_id, attempt_id = bound
        attributes[INTERGRAX_RUN_ID_ATTR] = str(run_id)
        attributes[INTERGRAX_ATTEMPT_ID_ATTR] = str(attempt_id)
    execution_id = peek_active_execution_id()
    if execution_id is not None:
        attributes[INTERGRAX_EXECUTION_ID_ATTR] = str(execution_id)
    return attributes


def merge_safe_span_attributes(
    *,
    caller_attributes: Mapping[str, SpanAttributeValue | None] | None = None,
    include_active_identity: bool = True,
) -> dict[str, SpanAttributeValue]:
    """Merge caller attributes with active identity, dropping unsafe raw-content keys."""
    merged: dict[str, SpanAttributeValue] = {}
    if include_active_identity:
        merged.update(active_execution_span_attributes())
    if caller_attributes:
        for key, value in caller_attributes.items():
            if value is None:
                continue
            if not is_safe_instrumentation_span_attribute_key(key):
                continue
            merged[key] = value
    return merged


def normalize_span_attribute_value(
    value: SpanAttributeValue | None,
) -> SpanAttributeValue | None:
    """Return a scalar span attribute value or ``None`` when the attribute should be omitted."""
    return value


def is_safe_instrumentation_span_attribute_key(key: str) -> bool:
    """Return whether a span attribute key is safe to emit by default."""
    lowered = key.strip().lower()
    if not lowered:
        return False
    if lowered.endswith((".length", ".count", "_length", "_count", "_bytes", "_ms")):
        return True
    if lowered.endswith(".query") or lowered == "query":
        return False
    return not any(fragment in lowered for fragment in _UNSAFE_ATTRIBUTE_KEY_FRAGMENTS)
