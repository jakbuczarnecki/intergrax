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
from intergrax.contracts.marketplace.diagnostics import (
    MarketplaceDiagnosticEvent,
    MarketplaceDiagnosticEventKind,
    MarketplaceDiagnosticOutcome,
    MarketplacePipelineStage,
)
from intergrax.marketplace.diagnostics import emit_marketplace_diagnostic
from intergrax.marketplace.diagnostics.session import MarketplacePipelineObservationSession


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
        observation: MarketplacePipelineObservationSession | None = None,
    ) -> tuple[RankedCapabilityCandidate, ...]:
        input_count = len(candidates)
        searched = search_capability_candidates(
            candidates,
            self.search_strategy,
            query=search_query,
        )
        if observation is not None:
            emit_marketplace_diagnostic(
                observation,
                MarketplaceDiagnosticEvent(
                    stage=MarketplacePipelineStage.SEARCH,
                    event_kind=MarketplaceDiagnosticEventKind.COMPLETED,
                    correlation=observation.correlation,
                    strategy_id=self.search_strategy.search_strategy_id,
                    input_count=input_count,
                    output_count=len(searched),
                    outcome=(
                        MarketplaceDiagnosticOutcome.EMPTY
                        if not searched
                        else MarketplaceDiagnosticOutcome.SUCCESS
                    ),
                ),
            )
        searched_candidates = tuple(item.candidate for item in searched)
        ranked = rank_capability_candidates(
            searched_candidates,
            self.ranker,
            context=ranking_context,
        )
        if observation is not None:
            ranker_id = self.ranker.ranker_id
            ranker_id_from_evidence = (
                ranked[0].evidence.ranker_id if ranked else ranker_id
            )
            emit_marketplace_diagnostic(
                observation,
                MarketplaceDiagnosticEvent(
                    stage=MarketplacePipelineStage.RANKING,
                    event_kind=MarketplaceDiagnosticEventKind.COMPLETED,
                    correlation=observation.correlation,
                    ranker_id=ranker_id_from_evidence,
                    input_count=len(searched_candidates),
                    output_count=len(ranked),
                    outcome=(
                        MarketplaceDiagnosticOutcome.EMPTY
                        if not ranked
                        else MarketplaceDiagnosticOutcome.SUCCESS
                    ),
                ),
            )
        return ranked
