# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable capability ranking over Stage-3 discovery candidates (Stage 4)."""

from __future__ import annotations

from typing import Final, Protocol

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
from intergrax.capability_catalog.ranking_validation import validate_ranked_output
from intergrax.contracts.capability_catalog.ranking import (
    CapabilityRankingContext,
    CapabilityRankingEvidence,
    CapabilityRankingSignal,
)

STABLE_IDENTITY_RANKER_ID: Final = "stable.identity"


class CapabilityRanker(Protocol):
    """Structural ranking plugin — ordering only, never selection."""

    @property
    def ranker_id(self) -> str:
        """Stable ranker identifier."""

    def rank(
        self,
        candidates: tuple[CapabilityDiscoveryCandidate, ...],
        context: CapabilityRankingContext,
    ) -> tuple[RankedCapabilityCandidate, ...]:
        """Reorder candidates and attach ranking evidence."""


class StableIdentityRanker:
    """Deterministic baseline — canonical identity order, not semantic relevance."""

    @property
    def ranker_id(self) -> str:
        return STABLE_IDENTITY_RANKER_ID

    def rank(
        self,
        candidates: tuple[CapabilityDiscoveryCandidate, ...],
        context: CapabilityRankingContext,
    ) -> tuple[RankedCapabilityCandidate, ...]:
        indexed = tuple((index, candidate) for index, candidate in enumerate(candidates))
        ordered = sorted(
            indexed,
            key=lambda pair: (pair[1].identity.sort_key, pair[0]),
        )
        return tuple(
            RankedCapabilityCandidate(
                candidate=candidate,
                evidence=CapabilityRankingEvidence(
                    ranker_id=self.ranker_id,
                    rank_position=position,
                    signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
                    original_stage3_position=original_index + 1,
                ),
            )
            for position, (original_index, candidate) in enumerate(ordered, start=1)
        )


def rank_capability_candidates(
    candidates: tuple[CapabilityDiscoveryCandidate, ...],
    ranker: CapabilityRanker,
    *,
    context: CapabilityRankingContext | None = None,
) -> tuple[RankedCapabilityCandidate, ...]:
    """Run a ranker and enforce output integrity fail-closed."""
    ranking_context = context or CapabilityRankingContext()
    ranked = ranker.rank(candidates, ranking_context)
    validate_ranked_output(
        input_candidates=candidates,
        ranked=ranked,
        ranker_id=ranker.ranker_id,
    )
    return ranked


def identity_sort_key(candidate: CapabilityDiscoveryCandidate) -> tuple[str, str, str, str]:
    """Public tie-break primitive aligned with Stage-1 identity semantics."""
    return candidate.identity.sort_key
