# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ranked discovery candidate projection (CAPABILITY-CATALOG-1 Stage 4)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.identity import CapabilityDiscoveryIdentity
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance
from intergrax.contracts.capability_catalog.ranking import CapabilityRankingEvidence

SCHEMA_RANKED_CAPABILITY_CANDIDATE_V1: Final = "ranked_capability_candidate.v1"


class RankedCapabilityCandidate(BaseModel):
    """Discovery candidate with typed ranking evidence — identity unchanged."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["ranked_capability_candidate.v1"] = (
        SCHEMA_RANKED_CAPABILITY_CANDIDATE_V1
    )
    candidate: CapabilityDiscoveryCandidate
    evidence: CapabilityRankingEvidence

    @property
    def identity(self) -> CapabilityDiscoveryIdentity:
        return self.candidate.identity

    @property
    def provenance(self) -> CapabilityProvenance:
        return self.candidate.provenance

    @property
    def availability(self) -> AvailabilityDisposition:
        return self.candidate.availability
