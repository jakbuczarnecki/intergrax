# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral agent project metadata provider port (AP-7)."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_EMPTY = Field(min_length=1)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class AgentProjectMetadata(BaseModel):
    """Parsed installed-agent project metadata — no filesystem access."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution_package_id: str = _NON_EMPTY
    dependencies: tuple[str, ...] = ()

    @field_validator("distribution_package_id")
    @classmethod
    def _strip_distribution_package_id(cls, value: str) -> str:
        return _strip_required(value)


class AgentProjectMetadataProvider(Protocol):
    """Resolve authoritative installed-agent metadata by opaque ref."""

    def get_metadata(self, metadata_ref: str) -> AgentProjectMetadata | None:
        """Load parsed metadata for one installed agent artifact."""
