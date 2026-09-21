# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral application observability attributes contract (OBS-EXPORT-4A)."""

from __future__ import annotations

from typing import Literal, Mapping, TypeAlias, cast

from pydantic import BaseModel, ConfigDict

APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA = "application_observability_attributes.v1"

ObservabilityAttributeValue: TypeAlias = str | int | float | bool | None | list[str]

_RESERVED_ATTRIBUTE_FIELDS: frozenset[str] = frozenset({"schema_version", "namespace"})

_UNSAFE = object()


def observability_attribute_key(namespace: str, field_name: str) -> str:
    """Return a stable namespaced export key for an application attribute field."""
    return f"{namespace}.{field_name}"


def _coerce_safe_attribute_value(value: object) -> ObservabilityAttributeValue | object:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        if all(isinstance(item, str) for item in value):
            return list(value)
        return _UNSAFE
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return value
        return _UNSAFE
    return _UNSAFE


class ApplicationObservabilityAttributes(BaseModel):
    """Base typed contract for safe application-specific observability metadata."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["application_observability_attributes.v1"] = (
        APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA
    )
    namespace: str
    operation: str | None = None

    def to_safe_attributes(self) -> Mapping[str, ObservabilityAttributeValue]:
        """Export declared fields as namespaced safe scalar/list attributes."""
        exported: dict[str, ObservabilityAttributeValue] = {}
        for field_name, value in self.model_dump(exclude_none=True).items():
            if field_name in _RESERVED_ATTRIBUTE_FIELDS:
                continue
            safe_value = _coerce_safe_attribute_value(value)
            if safe_value is _UNSAFE:
                continue
            exported[observability_attribute_key(self.namespace, field_name)] = cast(
                ObservabilityAttributeValue,
                safe_value,
            )

        exported[observability_attribute_key(self.namespace, "namespace")] = (
            self.namespace
        )
        if self.operation is not None:
            exported[observability_attribute_key(self.namespace, "operation")] = (
                self.operation
            )
        return exported


def coerce_observability_attribute_mapping(
    source: Mapping[str, object],
) -> dict[str, ObservabilityAttributeValue]:
    """Drop vendor fields that are not safe scalar/list observability attribute values."""
    exported: dict[str, ObservabilityAttributeValue] = {}
    for key, value in source.items():
        if not isinstance(key, str) or not key:
            continue
        safe = _coerce_safe_attribute_value(value)
        if safe is _UNSAFE:
            continue
        exported[key] = cast(ObservabilityAttributeValue, safe)
    return exported


__all__ = [
    "APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA",
    "ApplicationObservabilityAttributes",
    "ObservabilityAttributeValue",
    "coerce_observability_attribute_mapping",
    "observability_attribute_key",
]
