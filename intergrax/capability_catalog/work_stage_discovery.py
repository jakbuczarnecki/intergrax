# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Adaptive work-stage capability discovery (CAPABILITY-CATALOG-1 Stage 8)."""

from __future__ import annotations

from intergrax.capability_catalog.discovery import discover_capability_candidates
from intergrax.capability_catalog.governed_candidate import GovernedCapabilityCandidate
from intergrax.capability_catalog.governance import (
    CapabilityGovernanceEvaluator,
    govern_capability_candidates,
)
from intergrax.capability_catalog.ranking import CapabilityRanker, StableIdentityRanker
from intergrax.capability_catalog.snapshot import CapabilityCatalogSnapshot
from intergrax.capability_catalog.work_stage_effective import (
    EffectiveCapabilitySet,
    WorkStageCapabilityDiscoveryEvidence,
)
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)
from intergrax.contracts.capability_catalog.governance import CapabilityGovernanceContext
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
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
                need=need,
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
        effective_candidates = _select_effective_executable_candidates(
            governed_result.allowed,
        )
        catalog_only_keys = _catalog_only_allowed_keys(
            governed_result.allowed,
            effective_candidates,
        )
        effective_set = EffectiveCapabilitySet(
            need=need,
            governed_result=governed_result,
            effective_candidates=effective_candidates,
        )
        return WorkStageCapabilityDiscoveryEvidence(
            need=need,
            effective_set=effective_set,
            catalog_only_identity_keys=catalog_only_keys,
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


def _select_effective_executable_candidates(
    allowed: tuple[GovernedCapabilityCandidate, ...],
) -> tuple[GovernedCapabilityCandidate, ...]:
    executable = tuple(
        candidate
        for candidate in allowed
        if candidate.availability is AvailabilityDisposition.HOST_AVAILABLE
    )
    return tuple(
        sorted(
            executable,
            key=lambda candidate: candidate.ranked.identity.sort_key,
        )
    )


def _catalog_only_allowed_keys(
    allowed: tuple[GovernedCapabilityCandidate, ...],
    effective: tuple[GovernedCapabilityCandidate, ...],
) -> tuple[CapabilityIdentityKey, ...]:
    effective_keys = frozenset(
        candidate.ranked.identity.sort_key for candidate in effective
    )
    catalog_only = tuple(
        CapabilityIdentityKey.from_discovery_identity(candidate.identity)
        for candidate in allowed
        if candidate.availability is AvailabilityDisposition.CATALOG_AVAILABLE
        and candidate.ranked.identity.sort_key not in effective_keys
    )
    return tuple(sorted(catalog_only, key=lambda key: key.sort_key))
