# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed federated catalog entry projection (CAPABILITY-CATALOG-1 Stage 2)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog.identity import CapabilityDiscoveryIdentity
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance

SCHEMA_CAPABILITY_CATALOG_ENTRY_V1: Final = "capability_catalog_entry.v1"

_NON_EMPTY = Field(min_length=1)


class CapabilityCatalogEntry(BaseModel):
    """Read-only federated catalog entry — identity and provenance from Stage 1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_catalog_entry.v1"] = SCHEMA_CAPABILITY_CATALOG_ENTRY_V1
    identity: CapabilityDiscoveryIdentity
    provenance: CapabilityProvenance
    display_label: str | None = None

    @field_validator("display_label")
    @classmethod
    def _validate_display_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("display_label must be non-empty when provided")
        return normalized

    @model_validator(mode="after")
    def _validate_identity_provenance_source_consistency(self) -> CapabilityCatalogEntry:
        if self.identity.source != self.provenance.source:
            raise ValueError("identity.source must equal provenance.source")
        return self
