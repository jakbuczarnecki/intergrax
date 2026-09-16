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
from intergrax.contracts.marketplace.diagnostics import (
    MarketplaceDiagnosticEvent,
    MarketplaceDiagnosticEventKind,
    MarketplaceDiagnosticOutcome,
    MarketplacePipelineStage,
)
from intergrax.marketplace.diagnostics import emit_marketplace_diagnostic
from intergrax.marketplace.diagnostics.session import MarketplacePipelineObservationSession


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
        observation: MarketplacePipelineObservationSession | None = None,
    ) -> tuple[CapabilityRecommendation, ...]:
        input_count = len(governed_candidates)
        recommendations = recommend_capability_candidates(
            governed_candidates,
            self.recommendation_strategy,
            context=recommendation_context,
        )
        if observation is not None:
            emit_marketplace_diagnostic(
                observation,
                MarketplaceDiagnosticEvent(
                    stage=MarketplacePipelineStage.RECOMMENDATION,
                    event_kind=MarketplaceDiagnosticEventKind.COMPLETED,
                    correlation=observation.correlation,
                    recommendation_strategy_id=(
                        self.recommendation_strategy.recommendation_strategy_id
                    ),
                    input_count=input_count,
                    output_count=len(recommendations),
                    outcome=(
                        MarketplaceDiagnosticOutcome.EMPTY
                        if not recommendations
                        else MarketplaceDiagnosticOutcome.SUCCESS
                    ),
                ),
            )
        return recommendations
