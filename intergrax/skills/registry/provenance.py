# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime binding provenance for audit-grade exact skill release tracking."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_EMPTY = Field(min_length=1)

SCHEMA_SKILL_RUNTIME_BINDING_METADATA_V1: Final = (
    "skill_runtime_binding_metadata.v1"
)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class SkillRuntimeBindingMetadata(BaseModel):
    """Release identity retained on registry read surface after host binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_SKILL_RUNTIME_BINDING_METADATA_V1
    catalog_source_id: str = _NON_EMPTY
    logical_skill_id: str = _NON_EMPTY
    package_reference: str = _NON_EMPTY
    version_label: str = _NON_EMPTY
    content_digest: str = _NON_EMPTY

    @field_validator(
        "catalog_source_id",
        "logical_skill_id",
        "package_reference",
        "version_label",
        "content_digest",
    )
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)


__all__ = [
    "SCHEMA_SKILL_RUNTIME_BINDING_METADATA_V1",
    "SkillRuntimeBindingMetadata",
]
