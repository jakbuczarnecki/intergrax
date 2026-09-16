# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Thin marketplace discovery intelligence — search and rank via contracts only."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
from intergrax.capability_catalog.ranking import (
    CapabilityRanker,
    StableIdentityRanker,
    rank_capability_candidates,
)
from intergrax.capability_catalog.default_text_search import (
    DefaultCatalogEntryTextSearchStrategy,
)
from intergrax.capability_catalog.search import (
    CapabilitySearchStrategy,
    search_capability_candidates,
)
from intergrax.contracts.capability_catalog.ranking import CapabilityRankingContext
from intergrax.contracts.capability_catalog.search import CapabilitySearchQuery


@dataclass(frozen=True, slots=True)
class MarketplaceDiscoveryService:
    """Orchestrates search → rank without governance or recommendation."""

    search_strategy: CapabilitySearchStrategy
    ranker: CapabilityRanker

    @classmethod
    def with_defaults(cls) -> MarketplaceDiscoveryService:
        return cls(
            search_strategy=DefaultCatalogEntryTextSearchStrategy(),
            ranker=StableIdentityRanker(),
        )

    def search_and_rank(
        self,
        candidates: tuple[CapabilityDiscoveryCandidate, ...],
        *,
        search_query: CapabilitySearchQuery | None = None,
        ranking_context: CapabilityRankingContext | None = None,
    ) -> tuple[RankedCapabilityCandidate, ...]:
        searched = search_capability_candidates(
            candidates,
            self.search_strategy,
            query=search_query,
        )
        searched_candidates = tuple(item.candidate for item in searched)
        return rank_capability_candidates(
            searched_candidates,
            self.ranker,
            context=ranking_context,
        )
