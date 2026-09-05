# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Discovery candidate projection (CAPABILITY-CATALOG-1 Stage 3)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.identity import CapabilityDiscoveryIdentity
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance

SCHEMA_CAPABILITY_DISCOVERY_CANDIDATE_V1: Final = "capability_discovery_candidate.v1"


class CapabilityDiscoveryCandidate(BaseModel):
    """Read-only discovery candidate preserving Stage-1/2 identity and provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_discovery_candidate.v1"] = (
        SCHEMA_CAPABILITY_DISCOVERY_CANDIDATE_V1
    )
    catalog_entry: CapabilityCatalogEntry
    availability: AvailabilityDisposition

    @property
    def identity(self) -> CapabilityDiscoveryIdentity:
        return self.catalog_entry.identity

    @property
    def provenance(self) -> CapabilityProvenance:
        return self.catalog_entry.provenance
