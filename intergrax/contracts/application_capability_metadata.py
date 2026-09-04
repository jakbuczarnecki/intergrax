# © Artur Czarnecki. All rights reserved.

"""Neutral application capability metadata contracts for architecture discovery."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_EMPTY = Field(min_length=1)

SCHEMA_APPLICATION_CAPABILITY_DESCRIPTOR_V1: Final = "application_capability_descriptor.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class ApplicationCapabilityProjectionConflict(ValueError):
    """Raised when manifest projection cannot be mapped to capability identity."""


class ApplicationCapabilityDescriptor(BaseModel):
    """Declared application composition metadata for architecture discovery only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_APPLICATION_CAPABILITY_DESCRIPTOR_V1
    application_id: str = _NON_EMPTY
    application_version: str = _NON_EMPTY
    agent_contract_ids: tuple[str, ...] = ()
    default_capability: str | None = None

    @field_validator("application_id", "application_version")
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("default_capability")
    @classmethod
    def _strip_optional_default_capability(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @field_validator("agent_contract_ids")
    @classmethod
    def _strip_contract_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_strip_required(item) for item in value)


class ApplicationCapabilityMetadataProvider(Protocol):
    """Read-only port for non-executable application composition metadata."""

    def list_application_capability_descriptors(self) -> Sequence[ApplicationCapabilityDescriptor]:
        """List declared application metadata for architecture/discovery surfaces."""


def merge_application_capability_descriptors(
    descriptors: Sequence[ApplicationCapabilityDescriptor],
) -> tuple[ApplicationCapabilityDescriptor, ...]:
    """Merge descriptors with deterministic conflict semantics (identical rows dedupe)."""
    merged: dict[str, ApplicationCapabilityDescriptor] = {}
    for descriptor in descriptors:
        existing = merged.get(descriptor.application_id)
        if existing is None:
            merged[descriptor.application_id] = descriptor
            continue
        if existing == descriptor:
            continue
        raise ApplicationCapabilityProjectionConflict(
            f"conflicting application capability metadata for application_id="
            f"{descriptor.application_id!r}: existing={existing!r} incoming={descriptor!r}",
        )
    return tuple(sorted(merged.values(), key=lambda item: item.application_id))
