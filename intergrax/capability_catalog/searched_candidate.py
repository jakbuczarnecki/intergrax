# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Search-filtered discovery candidate projection (ME-5)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.identity import CapabilityDiscoveryIdentity
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance
from intergrax.contracts.capability_catalog.search import CapabilitySearchEvidence

SCHEMA_SEARCHED_CAPABILITY_CANDIDATE_V1: Final = "searched_capability_candidate.v1"


class SearchedCapabilityCandidate(BaseModel):
    """Discovery candidate with typed search evidence — identity unchanged."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["searched_capability_candidate.v1"] = (
        SCHEMA_SEARCHED_CAPABILITY_CANDIDATE_V1
    )
    candidate: CapabilityDiscoveryCandidate
    evidence: CapabilitySearchEvidence

    @property
    def identity(self) -> CapabilityDiscoveryIdentity:
        return self.candidate.identity

    @property
    def provenance(self) -> CapabilityProvenance:
        return self.candidate.provenance

    @property
    def availability(self) -> AvailabilityDisposition:
        return self.candidate.availability
