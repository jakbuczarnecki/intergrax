# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Discovery → governance → ranking → explicit selection → handoff orchestration (ME-10)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.query import CapabilityDiscoveryQuery
from intergrax.contracts.capability_catalog.release_identity import CapabilityReleaseIdentity
from intergrax.contracts.marketplace.diagnostics import (
    MarketplaceDiagnosticEvent,
    MarketplaceDiagnosticEventKind,
    MarketplaceDiagnosticObserver,
    MarketplaceDiagnosticOutcome,
    MarketplaceObserverFailurePolicy,
    MarketplacePipelineStage,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityDiscoveryTraceFacts,
    CapabilityHandoffConsumerTarget,
    CapabilityHandoffEnvelope,
    CapabilityMarketplaceExplicitSelection,
    consumer_target_for_kind,
)
from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext
from intergrax.marketplace.diagnostics import emit_marketplace_diagnostic
from intergrax.marketplace.diagnostics.session import MarketplacePipelineObservationSession
from intergrax.marketplace.discovery import MarketplaceDiscoveryService
from intergrax.marketplace.handoff_traceability.delivery import CapabilityHandoffDeliveryService
from intergrax.marketplace.handoff_traceability.errors import MarketplaceHandoffSelectionError
from intergrax.marketplace.observed_pipeline import run_marketplace_intelligence_pipeline
from intergrax.marketplace.service import MarketplaceCatalogService
from intergrax.capability_catalog.governance import CapabilityGovernanceEvaluator
from intergrax.contracts.capability_catalog.governance import CapabilityGovernanceContext


@dataclass(frozen=True, slots=True)
class MarketplaceDiscoveryHandoffOrchestrator:
    """Runs visibility-bounded discovery pipeline and explicit selection handoff only."""

    catalog_service: MarketplaceCatalogService
    discovery_service: MarketplaceDiscoveryService
    governance_evaluators: tuple[CapabilityGovernanceEvaluator, ...]
    governance_context: CapabilityGovernanceContext
    delivery_service: CapabilityHandoffDeliveryService
    diagnostic_observer: MarketplaceDiagnosticObserver | None = None
    observer_failure_policy: MarketplaceObserverFailurePolicy = (
        MarketplaceObserverFailurePolicy.BEST_EFFORT
    )

    def execute_explicit_selection_handoff(
        self,
        *,
        discovery_query: CapabilityDiscoveryQuery,
        marketplace_query_context: MarketplaceQueryContext,
        selected_identity_key: CapabilityIdentityKey,
        consumer_target: CapabilityHandoffConsumerTarget,
        selector_id: str,
        discovery_correlation_id: str,
        selection_id: str,
        handoff_id: str,
        query_correlation_id: str | None = None,
        query_text: str | None = None,
        recorded_at: datetime | None = None,
    ):
        query_context = marketplace_query_context
        observation = MarketplacePipelineObservationSession.for_discovery(
            discovery_correlation_id,
            query_correlation_id=query_correlation_id,
            observer=self.diagnostic_observer,
            failure_policy=self.observer_failure_policy,
        )
        pipeline = run_marketplace_intelligence_pipeline(
            catalog_service=self.catalog_service,
            discovery_service=self.discovery_service,
            governance_evaluators=self.governance_evaluators,
            governance_context=self.governance_context,
            discovery_query=discovery_query,
            marketplace_query_context=query_context,
            query_text=query_text,
            observation=observation,
        )
        listing_views = pipeline.listing_views
        governed = pipeline.governed.allowed
        selected_governed = None
        listing_id: str | None = None
        for item in governed:
            key = CapabilityIdentityKey.from_discovery_identity(item.identity)
            if key == selected_identity_key:
                selected_governed = item
                break
        if selected_governed is None:
            if observation.observer is not None:
                emit_marketplace_diagnostic(
                    observation,
                    MarketplaceDiagnosticEvent(
                        stage=MarketplacePipelineStage.SELECTION,
                        event_kind=MarketplaceDiagnosticEventKind.REJECTED,
                        correlation=observation.correlation,
                        outcome=MarketplaceDiagnosticOutcome.REJECTED,
                        detail="selected capability not in governed visible set",
                    ),
                )
            raise MarketplaceHandoffSelectionError(
                "selected capability is not in the governed visible candidate set",
            )
        for view in listing_views:
            view_key = CapabilityIdentityKey.from_discovery_identity(
                view.listing.capability.identity,
            )
            if view_key == selected_identity_key:
                listing_id = view.listing.listing_id
                if listing_id is None:
                    listing_id = view.listing.capability.identity.logical.logical_id
                break
        entry = selected_governed.ranked.candidate.catalog_entry
        selected_release = CapabilityReleaseIdentity.from_catalog_entry(entry)
        expected_target = consumer_target_for_kind(entry.identity.kind)
        if consumer_target is not expected_target:
            raise MarketplaceHandoffSelectionError(
                "consumer_target must align with selected capability kind",
            )
        ranking_evidence = selected_governed.ranking_evidence
        governance_evidence = selected_governed.evidence[0] if selected_governed.evidence else None
        trace = CapabilityDiscoveryTraceFacts(
            discovery_correlation_id=discovery_correlation_id,
            query_correlation_id=query_correlation_id,
            marketplace_query_context=query_context,
            visible_candidate_count=len(pipeline.listing_views),
            governed_admissible_count=len(governed),
            ranking_strategy_id=ranking_evidence.ranker_id,
            governance_evaluator_ids=tuple(
                sorted({item.evaluator_id for item in selected_governed.evidence}),
            ),
        )
        explicit_selection = CapabilityMarketplaceExplicitSelection(
            selection_id=selection_id,
            discovery_correlation_id=discovery_correlation_id,
            selected_release=selected_release,
            selector_id=selector_id,
            listing_id=listing_id,
            governance_evidence_ref=(
                governance_evidence.evaluator_id if governance_evidence is not None else None
            ),
            ranking_evidence_ref=ranking_evidence.ranker_id,
        )
        emit_marketplace_diagnostic(
            observation,
            MarketplaceDiagnosticEvent(
                stage=MarketplacePipelineStage.SELECTION,
                event_kind=MarketplaceDiagnosticEventKind.COMPLETED,
                correlation=observation.correlation,
                selected_release=selected_release,
                outcome=MarketplaceDiagnosticOutcome.SUCCESS,
            ),
        )
        timestamp = recorded_at or datetime.now(timezone.utc)
        envelope = CapabilityHandoffEnvelope(
            handoff_id=handoff_id,
            tenant_id=query_context.tenant_id,
            selected_release=selected_release,
            discovery_correlation_id=discovery_correlation_id,
            selection_id=selection_id,
            consumer_target=consumer_target,
            downstream_consumer_id=self.delivery_service.consumer.consumer_id,
            discovery_trace=trace,
            explicit_selection=explicit_selection,
            recorded_at=timestamp,
        )
        emit_marketplace_diagnostic(
            observation,
            MarketplaceDiagnosticEvent(
                stage=MarketplacePipelineStage.HANDOFF,
                event_kind=MarketplaceDiagnosticEventKind.STARTED,
                correlation=observation.correlation,
                selected_release=selected_release,
            ),
        )
        result = self.delivery_service.deliver(envelope)
        emit_marketplace_diagnostic(
            observation,
            MarketplaceDiagnosticEvent(
                stage=MarketplacePipelineStage.HANDOFF,
                event_kind=MarketplaceDiagnosticEventKind.COMPLETED,
                correlation=observation.correlation,
                selected_release=selected_release,
                handoff_status=result.disposition.value,
                outcome=MarketplaceDiagnosticOutcome.SUCCESS,
            ),
        )
        return result


__all__ = ["MarketplaceDiscoveryHandoffOrchestrator"]
