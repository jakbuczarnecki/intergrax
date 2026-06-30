# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed application observability attributes contract (OBS-EXPORT-4A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA = "application_observability_attributes.v1"
OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA = "observability_artifact_reference.v1"
SANITIZED_APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA = (
    "sanitized_application_observability_attributes.v1"
)

ObservabilityAttributeValue: TypeAlias = str | int | float | bool | None | list[str]

_RESERVED_ATTRIBUTE_FIELDS: frozenset[str] = frozenset({"schema_version", "namespace"})


def observability_attribute_key(namespace: str, field_name: str) -> str:
    """Return a stable namespaced export key for an application attribute field."""
    return f"{namespace}.{field_name}"


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
            exported[observability_attribute_key(self.namespace, field_name)] = safe_value

        exported[observability_attribute_key(self.namespace, "namespace")] = self.namespace
        if self.operation is not None:
            exported[observability_attribute_key(self.namespace, "operation")] = self.operation
        return exported


class ObservabilityArtifactReference(BaseModel):
    """Typed, reference-only artifact metadata for observability export envelopes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["observability_artifact_reference.v1"] = (
        OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA
    )
    artifact_ref: str = ""
    sha256: str = ""
    safe_relative_path: str = ""
    schema_id: str = ""

    @field_validator("artifact_ref", "safe_relative_path")
    @classmethod
    def _validate_relative_safe_path_fields(cls, value: str) -> str:
        if value and _looks_like_unsafe_path(value):
            raise ValueError("path fields must be relative-safe and must not contain path traversal")
        return value

    @model_validator(mode="after")
    def _validate_at_least_one_reference_field(self) -> ObservabilityArtifactReference:
        if not any((self.artifact_ref, self.sha256, self.safe_relative_path, self.schema_id)):
            raise ValueError(
                "At least one of artifact_ref, sha256, safe_relative_path, or schema_id must be present"
            )
        return self


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


_UNSAFE = object()


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


def _forbidden_attribute_field_names() -> frozenset[str]:
    from intergrax.runtime.observability.export_boundary import FORBIDDEN_EXPORT_CONTENT_FIELDS

    return FORBIDDEN_EXPORT_CONTENT_FIELDS


def _field_name_from_attribute_key(key: str) -> str:
    if "." not in key:
        return key
    return key.rsplit(".", 1)[-1]


def _looks_like_unsafe_path(value: str) -> bool:
    if not value:
        return False
    normalized = value.strip()
    if normalized.startswith(("/", "\\")):
        return True
    if len(normalized) > 1 and normalized[1] == ":" and normalized[0].isalpha():
        return True
    parts = normalized.replace("\\", "/").split("/")
    return ".." in parts


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
            and _looks_like_unsafe_path(safe_value)
        ):
            if hash_sensitive_paths:
                sanitized[key] = _hash_value(safe_value)
                hashed.append(key)
            else:
                dropped.append(key)
            continue

        sanitized[key] = safe_value

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
