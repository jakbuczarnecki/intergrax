# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable capability recommendation over ranked candidates (ME-5)."""

from __future__ import annotations

from typing import Final, Protocol

from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
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
        ranked: tuple[RankedCapabilityCandidate, ...],
        context: CapabilityRecommendationContext,
    ) -> tuple[CapabilityRecommendation, ...]:
        """Return advisory recommendations with explainability evidence."""


class DefaultTopRankedCapabilityRecommendationStrategy:
    """Deterministic baseline — first N ranked candidates, no semantic scoring."""

    @property
    def recommendation_strategy_id(self) -> str:
        return DEFAULT_TOP_RANKED_RECOMMENDATION_STRATEGY_ID

    def recommend(
        self,
        ranked: tuple[RankedCapabilityCandidate, ...],
        context: CapabilityRecommendationContext,
    ) -> tuple[CapabilityRecommendation, ...]:
        selected = ranked[: context.top_n]
        return tuple(
            CapabilityRecommendation(
                ranked=item,
                evidence=CapabilityRecommendationEvidence(
                    recommendation_strategy_id=self.recommendation_strategy_id,
                    reason_codes=(CapabilityRecommendationReasonCode.TOP_RANKED,),
                    reason_text="advisory top-ranked candidate",
                    rank_position=item.evidence.rank_position,
                ),
            )
            for item in selected
        )


def recommend_capability_candidates(
    ranked: tuple[RankedCapabilityCandidate, ...],
    strategy: CapabilityRecommendationStrategy,
    *,
    context: CapabilityRecommendationContext | None = None,
) -> tuple[CapabilityRecommendation, ...]:
    """Run a recommendation strategy and enforce output integrity fail-closed."""
    recommendation_context = context or CapabilityRecommendationContext()
    recommendations = strategy.recommend(ranked, recommendation_context)
    validate_recommendation_output(
        input_ranked=ranked,
        recommendations=recommendations,
        recommendation_strategy_id=strategy.recommendation_strategy_id,
    )
    return recommendations
