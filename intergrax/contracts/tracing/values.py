# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""JSON-safe trace attribute and payload value contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from intergrax.contracts.structured_json_value import (
    StructuredJsonObject,
    StructuredJsonPrimitive,
    StructuredJsonValue,
    normalize_structured_json_object,
    validate_structured_json_value,
)

TraceScalar: TypeAlias = StructuredJsonPrimitive
TraceValue: TypeAlias = StructuredJsonValue
TraceObject: TypeAlias = StructuredJsonObject

__all__ = [
    "TraceObject",
    "TraceScalar",
    "TraceValue",
    "normalize_trace_object",
    "normalize_trace_tags",
    "validate_trace_value",
]

validate_trace_value = validate_structured_json_value


def normalize_trace_tags(value: Mapping[str, object]) -> TraceObject:
    """Validate trace event tags and return an independent dict copy."""
    return normalize_structured_json_object(value, field_name="tags")


def normalize_trace_object(value: Mapping[str, object], *, field_name: str) -> TraceObject:
    """Validate a JSON object used in tool-call trace fields."""
    return normalize_structured_json_object(value, field_name=field_name)
