# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Deterministic JSON-safe structured values for platform contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum
from typing import TypeAlias

StructuredJsonPrimitive: TypeAlias = str | int | float | bool | None

type StructuredJsonValue = (
    StructuredJsonPrimitive | list[StructuredJsonValue] | dict[str, StructuredJsonValue]
)
StructuredJsonObject: TypeAlias = dict[str, StructuredJsonValue]


def validate_structured_json_value(
    value: object,
    *,
    field_name: str,
    path: str = "",
) -> StructuredJsonValue:
    """Reject non-JSON-safe values with an explicit, deterministic error."""
    if isinstance(value, Enum):
        label = path.rstrip(".") if path else field_name
        raise ValueError(
            f"{field_name} must contain JSON-compatible values at '{label}'"
        )

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            label = path.rstrip(".") if path else field_name
            raise ValueError(
                f"{field_name} must not contain non-finite float at '{label}'"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, StructuredJsonValue] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                child_path = path or field_name
                raise ValueError(f"{field_name} keys must be strings at '{child_path}'")
            result[raw_key] = validate_structured_json_value(
                child,
                field_name=field_name,
                path=f"{path}{raw_key}.",
            )
        return result
    if isinstance(value, list | tuple):
        return [
            validate_structured_json_value(
                child,
                field_name=field_name,
                path=f"{path}[{index}].",
            )
            for index, child in enumerate(value)
        ]

    label = path.rstrip(".") if path else field_name
    raise ValueError(f"{field_name} must contain JSON-compatible values at '{label}'")


def normalize_structured_json_object(
    value: Mapping[str, object],
    *,
    field_name: str,
) -> StructuredJsonObject:
    """Validate and return a plain dict copy suitable for contract storage."""
    validated = validate_structured_json_value(value, field_name=field_name)
    if not isinstance(validated, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return dict(validated)
