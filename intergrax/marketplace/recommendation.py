# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace recommendation orchestration over governance-admissible candidates."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.capability_catalog.governed_candidate import GovernedCapabilityCandidate
from intergrax.capability_catalog.recommendation import (
    CapabilityRecommendationStrategy,
    DefaultTopRankedCapabilityRecommendationStrategy,
    recommend_capability_candidates,
)
from intergrax.capability_catalog.recommended_capability import CapabilityRecommendation
from intergrax.contracts.capability_catalog.recommendation import (
    CapabilityRecommendationContext,
)


@dataclass(frozen=True, slots=True)
class MarketplaceRecommendationService:
    """Runs recommendation strategies on governed admissible input only."""

    recommendation_strategy: CapabilityRecommendationStrategy

    @classmethod
    def with_defaults(cls) -> MarketplaceRecommendationService:
        return cls(
            recommendation_strategy=DefaultTopRankedCapabilityRecommendationStrategy(),
        )

    def recommend(
        self,
        governed_candidates: tuple[GovernedCapabilityCandidate, ...],
        *,
        recommendation_context: CapabilityRecommendationContext | None = None,
    ) -> tuple[CapabilityRecommendation, ...]:
        return recommend_capability_candidates(
            governed_candidates,
            self.recommendation_strategy,
            context=recommendation_context,
        )
