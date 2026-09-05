# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Source-qualified discovery identity projections (CAPABILITY-CATALOG-1 Stage 1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.kind import CapabilityKind

_NON_EMPTY = Field(min_length=1)

SCHEMA_CAPABILITY_SOURCE_IDENTITY_V1: Final = "capability_source_identity.v1"
SCHEMA_CAPABILITY_LOGICAL_IDENTITY_V1: Final = "capability_logical_identity.v1"
SCHEMA_CAPABILITY_DISCOVERY_IDENTITY_V1: Final = "capability_discovery_identity.v1"


class CapabilityCatalogContractError(ValueError):
    """Malformed capability catalog contract."""


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

    schema_version: Literal["capability_source_identity.v1"] = (
        SCHEMA_CAPABILITY_SOURCE_IDENTITY_V1
    )
    source_id: str = _NON_EMPTY
    source_kind: CapabilitySourceKind = CapabilitySourceKind.UNKNOWN

    @field_validator("source_id")
    @classmethod
    def _validate_source_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="source_id")


class CapabilityLogicalIdentity(BaseModel):
    """Domain-native logical identity projection — read/discovery only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_logical_identity.v1"] = (
        SCHEMA_CAPABILITY_LOGICAL_IDENTITY_V1
    )
    kind: CapabilityKind
    logical_id: str = _NON_EMPTY

    @field_validator("logical_id")
    @classmethod
    def _validate_logical_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="logical_id")


class CapabilityDiscoveryIdentity(BaseModel):
    """Source-qualified discovery identity — not domain lifecycle authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_discovery_identity.v1"] = (
        SCHEMA_CAPABILITY_DISCOVERY_IDENTITY_V1
    )
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
