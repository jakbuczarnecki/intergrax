# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable capability recommendation over governance-admissible candidates (ME-5 / ME-5-C1)."""

from __future__ import annotations

from typing import Final, Protocol

from intergrax.capability_catalog.errors import CapabilityRecommendationError
from intergrax.capability_catalog.governed_candidate import GovernedCapabilityCandidate
from intergrax.capability_catalog.recommendation_validation import (
    validate_recommendation_output,
)
from intergrax.capability_catalog.recommended_capability import CapabilityRecommendation
from intergrax.contracts.capability_catalog.recommendation import (
    CapabilityRecommendationContext,
    CapabilityRecommendationEvidence,
    CapabilityRecommendationReasonCode,
)

DEFAULT_TOP_RANKED_RECOMMENDATION_STRATEGY_ID: Final = "default.top_ranked"


class CapabilityRecommendationStrategy(Protocol):
    """Structural recommendation plugin — advisory only, never lifecycle."""

    @property
    def recommendation_strategy_id(self) -> str:
        """Stable recommendation strategy identifier."""

    def recommend(
        self,
        governed: tuple[GovernedCapabilityCandidate, ...],
        context: CapabilityRecommendationContext,
    ) -> tuple[CapabilityRecommendation, ...]:
        """Return advisory recommendations with explainability evidence."""


class DefaultTopRankedCapabilityRecommendationStrategy:
    """Deterministic baseline — first N governed candidates, preserves rank order."""

    @property
    def recommendation_strategy_id(self) -> str:
        return DEFAULT_TOP_RANKED_RECOMMENDATION_STRATEGY_ID

    def recommend(
        self,
        governed: tuple[GovernedCapabilityCandidate, ...],
        context: CapabilityRecommendationContext,
    ) -> tuple[CapabilityRecommendation, ...]:
        selected = governed[: context.top_n]
        return tuple(
            CapabilityRecommendation(
                governed=item,
                evidence=CapabilityRecommendationEvidence(
                    recommendation_strategy_id=self.recommendation_strategy_id,
                    reason_codes=(CapabilityRecommendationReasonCode.TOP_RANKED,),
                    reason_text="advisory top-ranked admissible candidate",
                    rank_position=item.ranking_evidence.rank_position,
                ),
            )
            for item in selected
        )


def recommend_capability_candidates(
    governed: tuple[GovernedCapabilityCandidate, ...],
    strategy: CapabilityRecommendationStrategy,
    *,
    context: CapabilityRecommendationContext | None = None,
) -> tuple[CapabilityRecommendation, ...]:
    """Run a recommendation strategy on admissible governed input (fail-closed)."""
    for item in governed:
        if not isinstance(item, GovernedCapabilityCandidate):
            raise CapabilityRecommendationError(
                "recommendation input must be governance-admissible "
                "GovernedCapabilityCandidate instances",
            )
    recommendation_context = context or CapabilityRecommendationContext()
    recommendations = strategy.recommend(governed, recommendation_context)
    validate_recommendation_output(
        input_governed=governed,
        recommendations=recommendations,
        recommendation_strategy_id=strategy.recommendation_strategy_id,
    )
    return recommendations
