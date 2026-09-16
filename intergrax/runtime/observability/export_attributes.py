# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed application observability attributes and export sanitization (OBS-EXPORT-4A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.application_observability_attributes import (
    APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA,
    ApplicationObservabilityAttributes,
    ObservabilityAttributeValue,
    _coerce_safe_attribute_value,
    observability_attribute_key,
)
from intergrax.contracts.observability_artifact_reference import (
    OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA,
    ObservabilityArtifactReference,
    looks_like_unsafe_observability_path,
)

SANITIZED_APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA = (
    "sanitized_application_observability_attributes.v1"
)

_UNSAFE = object()


class SanitizedApplicationObservabilityAttributes(BaseModel):
    """Policy-sanitized, immutable application observability attributes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["sanitized_application_observability_attributes.v1"] = (
        SANITIZED_APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA
    )
    namespace: str = ""
    attributes: dict[str, ObservabilityAttributeValue] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ApplicationObservabilityAttributePolicyResult:
    sanitized: SanitizedApplicationObservabilityAttributes | None
    dropped_keys: tuple[str, ...] = ()
    hashed_keys: tuple[str, ...] = ()


def _forbidden_attribute_field_names() -> frozenset[str]:
    from intergrax.runtime.observability.export_boundary import (
        FORBIDDEN_EXPORT_CONTENT_FIELDS,
    )

    return FORBIDDEN_EXPORT_CONTENT_FIELDS


def _field_name_from_attribute_key(key: str) -> str:
    if "." not in key:
        return key
    return key.rsplit(".", 1)[-1]


def _hash_value(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_application_observability_attributes(
    attributes: ApplicationObservabilityAttributes,
    *,
    strict_redaction: bool = True,
    hash_sensitive_paths: bool = True,
) -> ApplicationObservabilityAttributePolicyResult:
    """Apply export-safe filtering to typed application attributes."""
    forbidden = _forbidden_attribute_field_names()
    dropped: list[str] = []
    hashed: list[str] = []
    sanitized: dict[str, ObservabilityAttributeValue] = {}

    for key, raw_value in attributes.to_safe_attributes().items():
        field_name = _field_name_from_attribute_key(key)
        if field_name in forbidden:
            dropped.append(key)
            continue

        safe_value = _coerce_safe_attribute_value(raw_value)
        if safe_value is _UNSAFE:
            dropped.append(key)
            continue

        if (
            strict_redaction
            and isinstance(safe_value, str)
            and looks_like_unsafe_observability_path(safe_value)
        ):
            if hash_sensitive_paths:
                sanitized[key] = _hash_value(safe_value)
                hashed.append(key)
            else:
                dropped.append(key)
            continue

        sanitized[key] = cast(ObservabilityAttributeValue, safe_value)

    if not sanitized:
        return ApplicationObservabilityAttributePolicyResult(
            sanitized=None,
            dropped_keys=tuple(dropped),
            hashed_keys=tuple(hashed),
        )

    return ApplicationObservabilityAttributePolicyResult(
        sanitized=SanitizedApplicationObservabilityAttributes(
            namespace=attributes.namespace,
            attributes=sanitized,
        ),
        dropped_keys=tuple(dropped),
        hashed_keys=tuple(hashed),
    )


def sanitized_application_attributes_are_content_safe(
    attributes: SanitizedApplicationObservabilityAttributes | None,
) -> bool:
    """Return False when sanitized attributes expose forbidden raw-content field names."""
    if attributes is None:
        return True

    forbidden = _forbidden_attribute_field_names()
    for key in attributes.attributes:
        if _field_name_from_attribute_key(key) in forbidden:
            return False
    return True


__all__ = [
    "APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA",
    "ApplicationObservabilityAttributePolicyResult",
    "ApplicationObservabilityAttributes",
    "OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA",
    "ObservabilityArtifactReference",
    "ObservabilityAttributeValue",
    "SANITIZED_APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA",
    "SanitizedApplicationObservabilityAttributes",
    "observability_attribute_key",
    "sanitize_application_observability_attributes",
    "sanitized_application_attributes_are_content_safe",
]
