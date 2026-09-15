# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference-only artifact metadata for observability and evidence contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA = "observability_artifact_reference.v1"


def looks_like_unsafe_observability_path(value: str) -> bool:
    if not value:
        return False
    normalized = value.strip()
    if normalized.startswith(("/", "\\")):
        return True
    if len(normalized) > 1 and normalized[1] == ":" and normalized[0].isalpha():
        return True
    parts = normalized.replace("\\", "/").split("/")
    return ".." in parts


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
        if value and looks_like_unsafe_observability_path(value):
            raise ValueError("path fields must be relative-safe and must not contain path traversal")
        return value

    @model_validator(mode="after")
    def _validate_at_least_one_reference_field(self) -> ObservabilityArtifactReference:
        if not any((self.artifact_ref, self.sha256, self.safe_relative_path, self.schema_id)):
            raise ValueError(
                "At least one of artifact_ref, sha256, safe_relative_path, or schema_id must be present",
            )
        return self


__all__ = [
    "OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA",
    "ObservabilityArtifactReference",
    "looks_like_unsafe_observability_path",
]
