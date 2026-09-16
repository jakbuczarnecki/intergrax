# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace intelligence pipeline with optional stage diagnostics (ME-10)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.governance import (
    CapabilityGovernanceEvaluator,
    govern_capability_candidates,
)
from intergrax.capability_catalog.governed_result import GovernedDiscoveryResult
from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
from intergrax.capability_catalog.recommended_capability import CapabilityRecommendation
from intergrax.capability_catalog.snapshot import CapabilityCatalogFederationCompleteness
from intergrax.contracts.capability_catalog.governance import CapabilityGovernanceContext
from intergrax.contracts.capability_catalog.query import CapabilityDiscoveryQuery
from intergrax.contracts.capability_catalog.recommendation import (
    CapabilityRecommendationContext,
)
from intergrax.contracts.marketplace.diagnostics import (
    MarketplaceDiagnosticEvent,
    MarketplaceDiagnosticEventKind,
    MarketplaceDiagnosticOutcome,
    MarketplacePipelineStage,
)
from intergrax.contracts.marketplace.listing import MarketplaceCapabilityListingView
from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext
from intergrax.marketplace.diagnostics import emit_marketplace_diagnostic
from intergrax.marketplace.diagnostics.session import MarketplacePipelineObservationSession
from intergrax.marketplace.discovery import MarketplaceDiscoveryService
from intergrax.marketplace.recommendation import MarketplaceRecommendationService
from intergrax.marketplace.service import MarketplaceCatalogService


@dataclass(frozen=True, slots=True)
class MarketplaceIntelligencePipelineResult:
    listing_views: tuple[MarketplaceCapabilityListingView, ...]
    ranked: tuple[RankedCapabilityCandidate, ...]
    governed: GovernedDiscoveryResult
    recommendations: tuple[CapabilityRecommendation, ...]
    catalog_federation_completeness: CapabilityCatalogFederationCompleteness


def run_marketplace_intelligence_pipeline(
    *,
    catalog_service: MarketplaceCatalogService,
    discovery_service: MarketplaceDiscoveryService,
    governance_evaluators: tuple[CapabilityGovernanceEvaluator, ...],
    governance_context: CapabilityGovernanceContext,
    discovery_query: CapabilityDiscoveryQuery,
    marketplace_query_context: MarketplaceQueryContext,
    query_text: str | None = None,
    observation: MarketplacePipelineObservationSession | None = None,
    recommendation_service: MarketplaceRecommendationService | None = None,
    recommendation_context: CapabilityRecommendationContext | None = None,
) -> MarketplaceIntelligencePipelineResult:
    """Visibility-bounded listing discovery → search/rank → govern → optional recommend."""
    if observation is not None:
        emit_marketplace_diagnostic(
            observation,
            MarketplaceDiagnosticEvent(
                stage=MarketplacePipelineStage.DISCOVERY,
                event_kind=MarketplaceDiagnosticEventKind.STARTED,
                correlation=observation.correlation,
            ),
        )

    listing_query = catalog_service.query_listings(
        discovery_query,
        marketplace_query_context=marketplace_query_context,
        query_text=query_text,
        observation=observation,
    )
    listing_views = listing_query.listing_views

    if observation is not None:
        emit_marketplace_diagnostic(
            observation,
            MarketplaceDiagnosticEvent(
                stage=MarketplacePipelineStage.DISCOVERY,
                event_kind=MarketplaceDiagnosticEventKind.COMPLETED,
                correlation=observation.correlation,
                output_count=len(listing_views),
                outcome=(
                    MarketplaceDiagnosticOutcome.EMPTY
                    if not listing_views
                    else MarketplaceDiagnosticOutcome.SUCCESS
                ),
            ),
        )

    candidates = tuple(
        CapabilityDiscoveryCandidate(
            catalog_entry=view.listing.capability,
            availability=view.availability,
        )
        for view in listing_views
    )
    ranked = discovery_service.search_and_rank(
        candidates,
        observation=observation,
    )
    governed = govern_capability_candidates(
        ranked,
        evaluators=governance_evaluators,
        context=governance_context,
    )
    if observation is not None:
        evaluator_ids = tuple(sorted({ev.evaluator_id for ev in governance_evaluators}))
        evidence_refs: list[str] = []
        for item in governed.allowed:
            for evidence in item.evidence:
                evidence_refs.append(evidence.evaluator_id)
        emit_marketplace_diagnostic(
            observation,
            MarketplaceDiagnosticEvent(
                stage=MarketplacePipelineStage.GOVERNANCE,
                event_kind=MarketplaceDiagnosticEventKind.COMPLETED,
                correlation=observation.correlation,
                governance_evaluator_ids=evaluator_ids,
                governance_evidence_refs=tuple(sorted(set(evidence_refs))),
                input_count=len(ranked),
                allowed_count=len(governed.allowed),
                blocked_count=len(governed.blocked),
                output_count=len(governed.allowed),
                outcome=(
                    MarketplaceDiagnosticOutcome.EMPTY
                    if not governed.allowed
                    else MarketplaceDiagnosticOutcome.SUCCESS
                ),
            ),
        )

    recommendations: tuple[CapabilityRecommendation, ...] = ()
    if recommendation_service is not None:
        recommendations = recommendation_service.recommend(
            governed.allowed,
            recommendation_context=recommendation_context,
            observation=observation,
        )

    return MarketplaceIntelligencePipelineResult(
        listing_views=listing_views,
        ranked=ranked,
        governed=governed,
        recommendations=recommendations,
        catalog_federation_completeness=listing_query.catalog_federation_completeness,
    )


__all__ = [
    "MarketplaceIntelligencePipelineResult",
    "run_marketplace_intelligence_pipeline",
]
