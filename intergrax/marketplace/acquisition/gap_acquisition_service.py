# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gap-anchored marketplace listing resolution and governed handoff (UCA-5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.capability_catalog.errors import CapabilityCatalogSourceFailure
from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.capability_catalog.governance import CapabilityGovernanceEvaluator
from intergrax.capability_catalog.recommended_capability import CapabilityRecommendation
from intergrax.contracts.capability_catalog.governance import (
    CapabilityGovernanceContext,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.recommendation import (
    CapabilityRecommendationContext,
)
from intergrax.contracts.capability_catalog.release_identity import (
    CapabilityReleaseIdentity,
)
from intergrax.contracts.marketplace.diagnostics import (
    MarketplaceDiagnosticObserver,
    MarketplaceObserverFailurePolicy,
)
from intergrax.contracts.marketplace.gap_acquisition import (
    MarketplaceGapAcquisitionOutcome,
    MarketplaceGapAcquisitionPort,
    MarketplaceGapAcquisitionRequest,
    MarketplaceGapAcquisitionResult,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerError,
    CapabilityHandoffConsumerFailureDisposition,
    CapabilityHandoffConsumerTarget,
    CapabilityHandoffDeliveryAdmissionError,
    CapabilityHandoffDeliveryDisposition,
    CapabilityHandoffDeliveryLifecycleTransitionError,
    CapabilityHandoffDeliveryOutcomeUncertainError,
    CapabilityHandoffIdentityConflictError,
    consumer_target_for_kind,
)
from intergrax.contracts.tools.marketplace_handoff_reference import (
    derive_marketplace_gap_tool_handoff_id,
    marketplace_domain_handoff_reference,
)
from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContextAssociationConflictError,
    MarketplaceQualifiedToolStageContextAssociationIntegrityError,
    MarketplaceQualifiedToolStageContextAssociationUnavailableError,
    MarketplaceQualifiedToolStageContextRecorder,
)
from intergrax.marketplace.acquisition.query_normalization import (
    discovery_query_for_machine_acquisition,
    effective_query_text,
)
from intergrax.marketplace.diagnostics.session import (
    MarketplacePipelineObservationSession,
)
from intergrax.marketplace.discovery import MarketplaceDiscoveryService
from intergrax.marketplace.handoff_traceability.errors import (
    MarketplaceHandoffSelectionError,
)
from intergrax.marketplace.handoff_traceability.orchestrator import (
    MarketplaceDiscoveryHandoffOrchestrator,
)
from intergrax.marketplace.observed_pipeline import (
    run_marketplace_intelligence_pipeline,
)
from intergrax.marketplace.recommendation import MarketplaceRecommendationService
from intergrax.marketplace.service import MarketplaceCatalogService


def _listing_correlation_id(operation_id: str) -> str:
    return f"marketplace-gap-listing:{operation_id}"


def _handoff_id(operation_id: str) -> str:
    return f"marketplace-gap-handoff:{operation_id}"


MARKETPLACE_GAP_SELECTION_ID_PREFIX = "marketplace-gap-selection:"


def marketplace_gap_selection_id(operation_id: str) -> str:
    normalized = require_non_empty_text(operation_id, label="operation_id")
    return f"{MARKETPLACE_GAP_SELECTION_ID_PREFIX}{normalized}"


def _selection_id(operation_id: str) -> str:
    return marketplace_gap_selection_id(operation_id)


def _marketplace_result_for_delivery_failure(
    *,
    request: MarketplaceGapAcquisitionRequest,
    listing_correlation_id: str,
    exc: BaseException,
    outcome: MarketplaceGapAcquisitionOutcome,
) -> MarketplaceGapAcquisitionResult:
    return MarketplaceGapAcquisitionResult(
        operation_id=request.operation_id,
        gap_id=request.gap_id,
        outcome=outcome,
        marketplace_listing_correlation_id=listing_correlation_id,
        reason_detail=str(exc),
    )


def _marketplace_outcome_for_consumer_failure(
    disposition: CapabilityHandoffConsumerFailureDisposition,
) -> MarketplaceGapAcquisitionOutcome:
    if disposition is CapabilityHandoffConsumerFailureDisposition.BLOCKED:
        return MarketplaceGapAcquisitionOutcome.BLOCKED
    if disposition is CapabilityHandoffConsumerFailureDisposition.UNAVAILABLE:
        return MarketplaceGapAcquisitionOutcome.UNAVAILABLE
    if disposition is CapabilityHandoffConsumerFailureDisposition.REQUIRES_HITL:
        return MarketplaceGapAcquisitionOutcome.REQUIRES_HITL
    if disposition is CapabilityHandoffConsumerFailureDisposition.FAILED:
        return MarketplaceGapAcquisitionOutcome.FAILED
    return MarketplaceGapAcquisitionOutcome.FAILED


def _top_recommendation(
    recommendations: tuple[CapabilityRecommendation, ...],
) -> CapabilityRecommendation | None:
    if not recommendations:
        return None
    return recommendations[0]


@dataclass(frozen=True, slots=True)
class MarketplaceGapAcquisitionService(MarketplaceGapAcquisitionPort):
    """Listing-source resolution and single-pass handoff for canonical CapabilityGap."""

    catalog_service: MarketplaceCatalogService
    discovery_service: MarketplaceDiscoveryService
    governance_evaluators: tuple[CapabilityGovernanceEvaluator, ...]
    governance_context: CapabilityGovernanceContext
    recommendation_service: MarketplaceRecommendationService
    handoff_orchestrator: MarketplaceDiscoveryHandoffOrchestrator
    recommendation_context: CapabilityRecommendationContext | None = None
    diagnostic_observer: MarketplaceDiagnosticObserver | None = None
    observer_failure_policy: MarketplaceObserverFailurePolicy = (
        MarketplaceObserverFailurePolicy.BEST_EFFORT
    )
    tool_stage_context_recorder: MarketplaceQualifiedToolStageContextRecorder | None = (
        None
    )

    def acquire_from_gap(
        self,
        request: MarketplaceGapAcquisitionRequest,
    ) -> MarketplaceGapAcquisitionResult:
        discovery_query = discovery_query_for_machine_acquisition(
            request.capability_need,
            request.discovery_query,
        )
        listing_correlation_id = _listing_correlation_id(request.operation_id)
        query_text = effective_query_text(
            request.capability_need,
            request.query_text,
        )
        observation = MarketplacePipelineObservationSession.for_discovery(
            listing_correlation_id,
            observer=self.diagnostic_observer,
            failure_policy=self.observer_failure_policy,
        )
        try:
            pipeline = run_marketplace_intelligence_pipeline(
                catalog_service=self.catalog_service,
                discovery_service=self.discovery_service,
                governance_evaluators=self.governance_evaluators,
                governance_context=self.governance_context,
                discovery_query=discovery_query,
                marketplace_query_context=request.marketplace_query_context,
                query_text=query_text,
                observation=observation,
                recommendation_service=self.recommendation_service,
                recommendation_context=self.recommendation_context,
            )
        except CapabilityCatalogSourceFailure as exc:
            return MarketplaceGapAcquisitionResult(
                operation_id=request.operation_id,
                gap_id=request.gap_id,
                outcome=MarketplaceGapAcquisitionOutcome.UNAVAILABLE,
                reason_detail=str(exc),
            )

        if not pipeline.listing_views:
            return MarketplaceGapAcquisitionResult(
                operation_id=request.operation_id,
                gap_id=request.gap_id,
                outcome=MarketplaceGapAcquisitionOutcome.NO_ACQUISITION_SOURCE,
                marketplace_listing_correlation_id=listing_correlation_id,
                reason_detail="no marketplace listing matches acquisition query",
            )
        if not pipeline.governed.allowed:
            return MarketplaceGapAcquisitionResult(
                operation_id=request.operation_id,
                gap_id=request.gap_id,
                outcome=MarketplaceGapAcquisitionOutcome.BLOCKED,
                marketplace_listing_correlation_id=listing_correlation_id,
                reason_detail="no governed marketplace acquisition source",
            )

        recommendation = _top_recommendation(pipeline.recommendations)
        if recommendation is None:
            return MarketplaceGapAcquisitionResult(
                operation_id=request.operation_id,
                gap_id=request.gap_id,
                outcome=MarketplaceGapAcquisitionOutcome.NO_ACQUISITION_SOURCE,
                marketplace_listing_correlation_id=listing_correlation_id,
                reason_detail="no marketplace recommendation for acquisition source",
            )

        release = CapabilityReleaseIdentity.from_catalog_entry(
            recommendation.governed.ranked.candidate.catalog_entry,
        )
        identity_key = CapabilityIdentityKey.from_discovery_identity(release.discovery)
        consumer_target = consumer_target_for_kind(release.discovery.kind)
        selection_id = _selection_id(request.operation_id)
        if consumer_target is CapabilityHandoffConsumerTarget.TOOL_DOMAIN:
            tenant_id = request.marketplace_query_context.tenant_id
            if tenant_id is None:
                return MarketplaceGapAcquisitionResult(
                    operation_id=request.operation_id,
                    gap_id=request.gap_id,
                    outcome=MarketplaceGapAcquisitionOutcome.BLOCKED,
                    marketplace_listing_correlation_id=listing_correlation_id,
                    reason_detail=(
                        "tenant_id required for qualified marketplace Tool handoff"
                    ),
                )
            handoff_id = derive_marketplace_gap_tool_handoff_id(
                tenant_id=tenant_id,
                operation_id=request.operation_id,
            )
            if self.tool_stage_context_recorder is not None:
                try:
                    self.tool_stage_context_recorder.record_tool_handoff_context(
                        handoff_id=handoff_id,
                        tenant_id=tenant_id,
                        acquisition_request_id=request.operation_id,
                    )
                except MarketplaceQualifiedToolStageContextAssociationConflictError as exc:
                    return MarketplaceGapAcquisitionResult(
                        operation_id=request.operation_id,
                        gap_id=request.gap_id,
                        outcome=MarketplaceGapAcquisitionOutcome.BLOCKED,
                        marketplace_listing_correlation_id=listing_correlation_id,
                        reason_detail=str(exc),
                    )
                except MarketplaceQualifiedToolStageContextAssociationUnavailableError as exc:
                    return MarketplaceGapAcquisitionResult(
                        operation_id=request.operation_id,
                        gap_id=request.gap_id,
                        outcome=MarketplaceGapAcquisitionOutcome.UNAVAILABLE,
                        marketplace_listing_correlation_id=listing_correlation_id,
                        reason_detail=str(exc),
                    )
                except MarketplaceQualifiedToolStageContextAssociationIntegrityError as exc:
                    return MarketplaceGapAcquisitionResult(
                        operation_id=request.operation_id,
                        gap_id=request.gap_id,
                        outcome=MarketplaceGapAcquisitionOutcome.FAILED,
                        marketplace_listing_correlation_id=listing_correlation_id,
                        reason_detail=str(exc),
                    )
        else:
            handoff_id = _handoff_id(request.operation_id)
        try:
            delivery = (
                self.handoff_orchestrator.deliver_explicit_selection_from_pipeline(
                    pipeline=pipeline,
                    marketplace_query_context=request.marketplace_query_context,
                    selected_identity_key=identity_key,
                    consumer_target=consumer_target,
                    selector_id=request.selector_id,
                    discovery_correlation_id=listing_correlation_id,
                    selection_id=selection_id,
                    handoff_id=handoff_id,
                    observation=observation,
                    recorded_at=datetime.now(tz=UTC),
                )
            )
        except MarketplaceHandoffSelectionError as exc:
            return MarketplaceGapAcquisitionResult(
                operation_id=request.operation_id,
                gap_id=request.gap_id,
                outcome=MarketplaceGapAcquisitionOutcome.BLOCKED,
                marketplace_listing_correlation_id=listing_correlation_id,
                reason_detail=str(exc),
            )
        except CapabilityHandoffConsumerError as exc:
            outcome = _marketplace_outcome_for_consumer_failure(exc.disposition)
            return MarketplaceGapAcquisitionResult(
                operation_id=request.operation_id,
                gap_id=request.gap_id,
                outcome=outcome,
                marketplace_listing_correlation_id=listing_correlation_id,
                reason_detail=str(exc),
            )
        except CapabilityHandoffDeliveryOutcomeUncertainError as exc:
            return _marketplace_result_for_delivery_failure(
                request=request,
                listing_correlation_id=listing_correlation_id,
                exc=exc,
                outcome=MarketplaceGapAcquisitionOutcome.FAILED,
            )
        except CapabilityHandoffDeliveryLifecycleTransitionError as exc:
            return _marketplace_result_for_delivery_failure(
                request=request,
                listing_correlation_id=listing_correlation_id,
                exc=exc,
                outcome=MarketplaceGapAcquisitionOutcome.FAILED,
            )
        except CapabilityHandoffIdentityConflictError as exc:
            return _marketplace_result_for_delivery_failure(
                request=request,
                listing_correlation_id=listing_correlation_id,
                exc=exc,
                outcome=MarketplaceGapAcquisitionOutcome.FAILED,
            )
        except CapabilityHandoffDeliveryAdmissionError as exc:
            return _marketplace_result_for_delivery_failure(
                request=request,
                listing_correlation_id=listing_correlation_id,
                exc=exc,
                outcome=MarketplaceGapAcquisitionOutcome.UNAVAILABLE,
            )

        if delivery.disposition not in (
            CapabilityHandoffDeliveryDisposition.DELIVERED,
            CapabilityHandoffDeliveryDisposition.DUPLICATE_SKIPPED,
        ):
            return MarketplaceGapAcquisitionResult(
                operation_id=request.operation_id,
                gap_id=request.gap_id,
                outcome=MarketplaceGapAcquisitionOutcome.FAILED,
                marketplace_listing_correlation_id=listing_correlation_id,
                reason_detail=f"unexpected handoff disposition: {delivery.disposition.value}",
            )

        return MarketplaceGapAcquisitionResult(
            operation_id=request.operation_id,
            gap_id=request.gap_id,
            outcome=MarketplaceGapAcquisitionOutcome.SUCCEEDED,
            marketplace_listing_correlation_id=listing_correlation_id,
            domain_handoff_reference=marketplace_domain_handoff_reference(handoff_id),
        )


__all__ = [
    "MARKETPLACE_GAP_SELECTION_ID_PREFIX",
    "MarketplaceGapAcquisitionService",
    "marketplace_gap_selection_id",
]
