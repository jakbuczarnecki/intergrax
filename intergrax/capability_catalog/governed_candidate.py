# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Governed discovery candidate projections (CAPABILITY-CATALOG-1 Stage 5)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.governance import (
    GovernanceDecisionEvidence,
    GovernanceDisposition,
)
from intergrax.contracts.capability_catalog.identity import CapabilityDiscoveryIdentity
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance
from intergrax.contracts.capability_catalog.ranking import CapabilityRankingEvidence

SCHEMA_GOVERNED_CAPABILITY_CANDIDATE_V1: Final = "governed_capability_candidate.v1"
SCHEMA_BLOCKED_CAPABILITY_CANDIDATE_V1: Final = "blocked_capability_candidate.v1"


class GovernedCapabilityCandidate(BaseModel):
    """Ranked candidate that passed governance narrowing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governed_capability_candidate.v1"] = (
        SCHEMA_GOVERNED_CAPABILITY_CANDIDATE_V1
    )
    ranked: RankedCapabilityCandidate
    evidence: tuple[GovernanceDecisionEvidence, ...]

    @model_validator(mode="after")
    def _validate_allowed_evidence(self) -> GovernedCapabilityCandidate:
        if not self.evidence:
            raise ValueError("governed candidate requires non-empty governance evidence")
        for item in self.evidence:
            if item.disposition is not GovernanceDisposition.ALLOWED:
                raise ValueError(
                    "governed candidate evidence must carry ALLOWED disposition",
                )
        return self

    @property
    def identity(self) -> CapabilityDiscoveryIdentity:
        return self.ranked.identity

    @property
    def provenance(self) -> CapabilityProvenance:
        return self.ranked.provenance

    @property
    def availability(self) -> AvailabilityDisposition:
        return self.ranked.availability

    @property
    def ranking_evidence(self) -> CapabilityRankingEvidence:
        return self.ranked.evidence


class BlockedCapabilityCandidate(BaseModel):
    """Ranked candidate blocked by governance narrowing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["blocked_capability_candidate.v1"] = (
        SCHEMA_BLOCKED_CAPABILITY_CANDIDATE_V1
    )
    ranked: RankedCapabilityCandidate
    evidence: tuple[GovernanceDecisionEvidence, ...]

    @model_validator(mode="after")
    def _validate_blocked_evidence(self) -> BlockedCapabilityCandidate:
        if not self.evidence:
            raise ValueError("blocked candidate requires non-empty governance evidence")
        if not any(
            item.disposition is GovernanceDisposition.BLOCKED for item in self.evidence
        ):
            raise ValueError(
                "blocked candidate evidence must include at least one BLOCKED disposition",
            )
        return self

    @property
    def identity(self) -> CapabilityDiscoveryIdentity:
        return self.ranked.identity

    @property
    def provenance(self) -> CapabilityProvenance:
        return self.ranked.provenance

    @property
    def availability(self) -> AvailabilityDisposition:
        return self.ranked.availability

    @property
    def ranking_evidence(self) -> CapabilityRankingEvidence:
        return self.ranked.evidence
