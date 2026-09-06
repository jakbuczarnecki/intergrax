# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Adaptive work-stage capability discovery (CAPABILITY-CATALOG-1 Stage 8)."""

from __future__ import annotations

from intergrax.capability_catalog.discovery import discover_capability_candidates
from intergrax.capability_catalog.governance import (
    CapabilityGovernanceEvaluator,
    govern_capability_candidates,
)
from intergrax.capability_catalog.ranking import CapabilityRanker, StableIdentityRanker
from intergrax.capability_catalog.snapshot import CapabilityCatalogSnapshot
from intergrax.capability_catalog.work_stage_effective import (
    EffectiveCapabilitySet,
    WorkStageCapabilityDiscoveryEvidence,
    select_effective_executable_candidates,
)
from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)
from intergrax.contracts.capability_catalog.governance import CapabilityGovernanceContext
from intergrax.contracts.capability_catalog.ranking import CapabilityRankingContext
from intergrax.contracts.capability_catalog.work_stage import WorkStageCapabilityNeed


class WorkStageCapabilityDiscoveryService:
    """Stateless resolver — rediscovery per stage, no hidden lifecycle state."""

    def __init__(
        self,
        *,
        governance_evaluators: tuple[CapabilityGovernanceEvaluator, ...],
        ranker: CapabilityRanker | None = None,
    ) -> None:
        self._governance_evaluators = governance_evaluators
        self._ranker = ranker or StableIdentityRanker()

    def resolve(
        self,
        need: WorkStageCapabilityNeed,
        *,
        snapshot: CapabilityCatalogSnapshot,
        availability_evidence: CapabilityDiscoveryAvailabilityEvidence,
        governance_context: CapabilityGovernanceContext,
    ) -> WorkStageCapabilityDiscoveryEvidence:
        """Discover, rank, govern, and narrow to executable effective capabilities."""
        if not need.requests_capabilities:
            empty_governed = govern_capability_candidates(
                (),
                evaluators=self._governance_evaluators,
                context=governance_context,
            )
            effective_set = EffectiveCapabilitySet(
                need=need,
                governed_result=empty_governed,
                effective_candidates=(),
            )
            return WorkStageCapabilityDiscoveryEvidence(
                effective_set=effective_set,
            )

        candidates = discover_capability_candidates(
            snapshot,
            need.discovery_query,
            availability_evidence=availability_evidence,
        )
        ranked = self._ranker.rank(
            candidates,
            CapabilityRankingContext(),
        )
        governed_result = govern_capability_candidates(
            ranked,
            evaluators=self._governance_evaluators,
            context=governance_context,
        )
        effective_candidates = select_effective_executable_candidates(
            governed_result.allowed,
        )
        effective_set = EffectiveCapabilitySet(
            need=need,
            governed_result=governed_result,
            effective_candidates=effective_candidates,
        )
        return WorkStageCapabilityDiscoveryEvidence(
            effective_set=effective_set,
        )


def discover_effective_capabilities_for_work_stage(
    need: WorkStageCapabilityNeed,
    *,
    snapshot: CapabilityCatalogSnapshot,
    availability_evidence: CapabilityDiscoveryAvailabilityEvidence,
    governance_context: CapabilityGovernanceContext,
    governance_evaluators: tuple[CapabilityGovernanceEvaluator, ...],
    ranker: CapabilityRanker | None = None,
) -> WorkStageCapabilityDiscoveryEvidence:
    """Module-level entry point wrapping ``WorkStageCapabilityDiscoveryService``."""
    service = WorkStageCapabilityDiscoveryService(
        governance_evaluators=governance_evaluators,
        ranker=ranker,
    )
    return service.resolve(
        need,
        snapshot=snapshot,
        availability_evidence=availability_evidence,
        governance_context=governance_context,
    )


