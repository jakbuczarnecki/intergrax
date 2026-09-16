# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime activation provenance for audit-grade exact release tracking."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_EMPTY = Field(min_length=1)

SCHEMA_TOOL_RUNTIME_ACTIVATION_METADATA_V1: Final = (
    "tool_runtime_activation_metadata.v1"
)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class ToolRuntimeActivationMetadata(BaseModel):
    """Release identity retained on registry read surface after activation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TOOL_RUNTIME_ACTIVATION_METADATA_V1
    catalog_source_id: str = _NON_EMPTY
    logical_tool_id: str = _NON_EMPTY
    package_reference: str = _NON_EMPTY
    version_label: str = _NON_EMPTY
    content_digest: str = _NON_EMPTY

    @field_validator(
        "catalog_source_id",
        "logical_tool_id",
        "package_reference",
        "version_label",
        "content_digest",
    )
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)


__all__ = [
    "SCHEMA_TOOL_RUNTIME_ACTIVATION_METADATA_V1",
    "ToolRuntimeActivationMetadata",
]
