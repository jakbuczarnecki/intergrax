# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Source-qualified discovery identity projections (CAPABILITY-CATALOG-1 Stage 1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import (
    normalize_optional_text,
    require_non_empty_text,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind

_NON_EMPTY = Field(min_length=1)

SCHEMA_CAPABILITY_SOURCE_IDENTITY_V1: Final = "capability_source_identity.v1"
SCHEMA_CAPABILITY_LOGICAL_IDENTITY_V1: Final = "capability_logical_identity.v1"
SCHEMA_CAPABILITY_DISCOVERY_IDENTITY_V1: Final = "capability_discovery_identity.v1"


class CapabilityCatalogContractError(ValueError):
    """Malformed capability catalog contract."""


class CapabilityDiscoveryIdentityConflict(CapabilityCatalogContractError):
    """Duplicate or conflicting source-qualified discovery identities."""


class CapabilitySourceKind(StrEnum):
    """Neutral catalog source classifier — not domain registry authority."""

    BUILTIN = "builtin"
    LOCAL = "local"
    ENTERPRISE_PRIVATE = "enterprise_private"
    OFFICIAL = "official"
    THIRD_PARTY = "third_party"
    UNKNOWN = "unknown"


class CapabilitySourceIdentity(BaseModel):
    """Explicit catalog/discovery source identity shared across V1 capability types."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_CAPABILITY_SOURCE_IDENTITY_V1
    source_id: str = _NON_EMPTY
    source_kind: CapabilitySourceKind = CapabilitySourceKind.UNKNOWN

    @field_validator("source_id")
    @classmethod
    def _validate_source_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="source_id")


class CapabilityLogicalIdentity(BaseModel):
    """Domain-native logical identity projection — read/discovery only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_CAPABILITY_LOGICAL_IDENTITY_V1
    kind: CapabilityKind
    logical_id: str = _NON_EMPTY

    @field_validator("logical_id")
    @classmethod
    def _validate_logical_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="logical_id")


class CapabilityDiscoveryIdentity(BaseModel):
    """Source-qualified discovery identity — not domain lifecycle authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_CAPABILITY_DISCOVERY_IDENTITY_V1
    kind: CapabilityKind
    source: CapabilitySourceIdentity
    logical: CapabilityLogicalIdentity

    @model_validator(mode="after")
    def _validate_kind_alignment(self) -> CapabilityDiscoveryIdentity:
        if self.logical.kind is not self.kind:
            raise CapabilityCatalogContractError(
                "logical identity kind must match discovery identity kind",
            )
        return self

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        return (
            self.kind.value,
            self.source.source_id,
            self.source.source_kind.value,
            self.logical.logical_id,
        )


def discovery_identity_sort_key(
    identity: CapabilityDiscoveryIdentity,
) -> tuple[str, str, str, str]:
    return identity.sort_key


def normalize_discovery_identity_set(
    identities: tuple[CapabilityDiscoveryIdentity, ...],
) -> tuple[CapabilityDiscoveryIdentity, ...]:
    """Deterministic ordering with fail-closed duplicate and conflict detection."""
    ordered = tuple(sorted(identities, key=discovery_identity_sort_key))
    seen: dict[tuple[str, str, str, str], CapabilityDiscoveryIdentity] = {}
    for identity in ordered:
        key = identity.sort_key
        existing = seen.get(key)
        if existing is not None:
            raise CapabilityDiscoveryIdentityConflict(
                "duplicate source-qualified discovery identity",
            )
        seen[key] = identity
    return ordered
