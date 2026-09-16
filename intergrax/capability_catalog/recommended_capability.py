# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Advisory capability recommendation projection (ME-5)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.identity import CapabilityDiscoveryIdentity
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance
from intergrax.contracts.capability_catalog.recommendation import (
    CapabilityRecommendationEvidence,
)

SCHEMA_CAPABILITY_RECOMMENDATION_V1: Final = "capability_recommendation.v1"


class CapabilityRecommendation(BaseModel):
    """Ranked candidate with advisory recommendation evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_recommendation.v1"] = SCHEMA_CAPABILITY_RECOMMENDATION_V1
    ranked: RankedCapabilityCandidate
    evidence: CapabilityRecommendationEvidence

    @property
    def identity(self) -> CapabilityDiscoveryIdentity:
        return self.ranked.identity

    @property
    def provenance(self) -> CapabilityProvenance:
        return self.ranked.provenance

    @property
    def availability(self) -> AvailabilityDisposition:
        return self.ranked.availability
