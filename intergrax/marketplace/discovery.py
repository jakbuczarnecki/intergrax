# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Thin marketplace discovery pipeline — search, rank, recommend via contracts only."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.ranking import (
    CapabilityRanker,
    StableIdentityRanker,
    rank_capability_candidates,
)
from intergrax.capability_catalog.recommendation import (
    CapabilityRecommendationStrategy,
    DefaultTopRankedCapabilityRecommendationStrategy,
    recommend_capability_candidates,
)
from intergrax.capability_catalog.recommended_capability import CapabilityRecommendation
from intergrax.capability_catalog.default_text_search import (
    DefaultCatalogEntryTextSearchStrategy,
)
from intergrax.capability_catalog.search import (
    CapabilitySearchStrategy,
    search_capability_candidates,
)
from intergrax.contracts.capability_catalog.ranking import CapabilityRankingContext
from intergrax.contracts.capability_catalog.recommendation import (
    CapabilityRecommendationContext,
)
from intergrax.contracts.capability_catalog.search import CapabilitySearchQuery


@dataclass(frozen=True, slots=True)
class MarketplaceDiscoveryService:
    """Orchestrates search → rank → recommend without embedding algorithms."""

    search_strategy: CapabilitySearchStrategy
    ranker: CapabilityRanker
    recommendation_strategy: CapabilityRecommendationStrategy

    @classmethod
    def with_defaults(cls) -> MarketplaceDiscoveryService:
        return cls(
            search_strategy=DefaultCatalogEntryTextSearchStrategy(),
            ranker=StableIdentityRanker(),
            recommendation_strategy=DefaultTopRankedCapabilityRecommendationStrategy(),
        )

    def discover_recommendations(
        self,
        candidates: tuple[CapabilityDiscoveryCandidate, ...],
        *,
        search_query: CapabilitySearchQuery | None = None,
        ranking_context: CapabilityRankingContext | None = None,
        recommendation_context: CapabilityRecommendationContext | None = None,
    ) -> tuple[CapabilityRecommendation, ...]:
        searched = search_capability_candidates(
            candidates,
            self.search_strategy,
            query=search_query,
        )
        searched_candidates = tuple(item.candidate for item in searched)
        ranked = rank_capability_candidates(
            searched_candidates,
            self.ranker,
            context=ranking_context,
        )
        return recommend_capability_candidates(
            ranked,
            self.recommendation_strategy,
            context=recommendation_context,
        )
