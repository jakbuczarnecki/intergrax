# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Machine capability acquisition facade over the common marketplace engine (ME-12)."""

from __future__ import annotations

from dataclasses import dataclass
from intergrax.capability_catalog.governance import CapabilityGovernanceEvaluator
from intergrax.capability_catalog.recommended_capability import CapabilityRecommendation
from intergrax.capability_catalog.snapshot import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.governance import (
    CapabilityGovernanceContext,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.release_identity import (
    CapabilityReleaseIdentity,
)
from intergrax.contracts.marketplace.acquisition import (
    MachineCapabilityAcquisitionHandoffRequest,
    MachineCapabilityAcquisitionHandoffResponse,
    MachineCapabilityAcquisitionOutcome,
    MachineCapabilityAcquisitionPolicy,
    MachineCapabilityAcquisitionRequest,
    MachineCapabilityAcquisitionResponse,
    MachineCapabilityRecommendation,
    MachineCatalogFederationCompleteness,
)
from intergrax.contracts.marketplace.diagnostics import (
    MarketplaceDiagnosticObserver,
    MarketplaceObserverFailurePolicy,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    consumer_target_for_kind,
)
from intergrax.marketplace.acquisition.errors import (
    MachineCapabilityAcquisitionPolicyError,
    MachineCapabilityAcquisitionSelectionError,
)
from intergrax.marketplace.acquisition.observation_resolution import (
    resolve_acquisition_discovery_correlation,
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
    MarketplaceIntelligencePipelineResult,
    run_marketplace_intelligence_pipeline,
)
from intergrax.marketplace.recommendation import MarketplaceRecommendationService
from intergrax.marketplace.service import MarketplaceCatalogService


def _to_machine_recommendation(
    recommendation: CapabilityRecommendation,
) -> MachineCapabilityRecommendation:
    governed = recommendation.governed
    return MachineCapabilityRecommendation(
        release=CapabilityReleaseIdentity.from_catalog_entry(
            governed.ranked.candidate.catalog_entry,
        ),
        recommendation_evidence=recommendation.evidence,
        governance_evidence_refs=tuple(
            sorted({item.evaluator_id for item in governed.evidence}),
        ),
        ranking_strategy_id=governed.ranking_evidence.ranker_id,
    )


def _apply_acquisition_policy(
    policy: MachineCapabilityAcquisitionPolicy | None,
    recommendations: tuple[MachineCapabilityRecommendation, ...],
) -> tuple[MachineCapabilityRecommendation, ...]:
    if policy is None:
        return recommendations
    narrowed = policy.narrow_recommendations(recommendations)
    allowed = set(recommendations)
    for item in narrowed:
        if item not in allowed:
            raise MachineCapabilityAcquisitionPolicyError(
                "acquisition policy cannot widen recommendations",
            )
    return narrowed


def _federation_completeness(
    value: CapabilityCatalogFederationCompleteness,
) -> MachineCatalogFederationCompleteness:
    if value is CapabilityCatalogFederationCompleteness.PARTIAL:
        return MachineCatalogFederationCompleteness.PARTIAL
    return MachineCatalogFederationCompleteness.COMPLETE


def _outcome_for_pipeline(
    *,
    visible_count: int,
    governed_count: int,
    recommendation_count: int,
) -> MachineCapabilityAcquisitionOutcome:
    if visible_count == 0:
        return MachineCapabilityAcquisitionOutcome.NO_MATCH
    if governed_count == 0:
        return MachineCapabilityAcquisitionOutcome.NO_GOVERNED_MATCH
    if recommendation_count == 0:
        return MachineCapabilityAcquisitionOutcome.NO_RECOMMENDATIONS
    return MachineCapabilityAcquisitionOutcome.RECOMMENDATIONS_AVAILABLE


@dataclass(frozen=True, slots=True)
class MachineCapabilityAcquisitionService:
    """Contract-driven machine consumer entry point — orchestrates existing marketplace pipeline."""

    catalog_service: MarketplaceCatalogService
    discovery_service: MarketplaceDiscoveryService
    governance_evaluators: tuple[CapabilityGovernanceEvaluator, ...]
    governance_context: CapabilityGovernanceContext
    recommendation_service: MarketplaceRecommendationService
    handoff_orchestrator: MarketplaceDiscoveryHandoffOrchestrator
    acquisition_policy: MachineCapabilityAcquisitionPolicy | None = None
    diagnostic_observer: MarketplaceDiagnosticObserver | None = None
    observer_failure_policy: MarketplaceObserverFailurePolicy = (
        MarketplaceObserverFailurePolicy.BEST_EFFORT
    )

    def acquire(
        self,
        request: MachineCapabilityAcquisitionRequest,
    ) -> MachineCapabilityAcquisitionResponse:
        response, _pipeline, _observation = self._acquire_with_pipeline(
            request,
            operation_discovery_correlation_id=None,
        )
        return response

    def _acquire(
        self,
        request: MachineCapabilityAcquisitionRequest,
        *,
        operation_discovery_correlation_id: str | None,
    ) -> MachineCapabilityAcquisitionResponse:
        response, _pipeline, _observation = self._acquire_with_pipeline(
            request,
            operation_discovery_correlation_id=operation_discovery_correlation_id,
        )
        return response

    def _acquire_with_pipeline(
        self,
        request: MachineCapabilityAcquisitionRequest,
        *,
        operation_discovery_correlation_id: str | None,
    ) -> tuple[
        MachineCapabilityAcquisitionResponse,
        MarketplaceIntelligencePipelineResult,
        MarketplacePipelineObservationSession,
    ]:
        discovery_query = discovery_query_for_machine_acquisition(
            request.need,
            request.discovery_query,
        )
        query_text = effective_query_text(request.need, request.query_text)
        discovery_correlation_id, query_correlation_id = (
            resolve_acquisition_discovery_correlation(
                request,
                operation_discovery_correlation_id=operation_discovery_correlation_id,
            )
        )
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
            marketplace_query_context=request.marketplace_query_context,
            query_text=query_text,
            observation=observation,
            recommendation_service=self.recommendation_service,
            recommendation_context=request.recommendation_context,
        )
        recommendations = _apply_acquisition_policy(
            self.acquisition_policy,
            tuple(
                _to_machine_recommendation(item) for item in pipeline.recommendations
            ),
        )
        outcome = _outcome_for_pipeline(
            visible_count=len(pipeline.listing_views),
            governed_count=len(pipeline.governed.allowed),
            recommendation_count=len(recommendations),
        )
        completeness = _federation_completeness(
            pipeline.catalog_federation_completeness
        )
        response = MachineCapabilityAcquisitionResponse(
            request_id=request.request_id,
            discovery_correlation_id=discovery_correlation_id,
            outcome=outcome,
            recommendations=recommendations,
            observation=request.observation or observation.correlation,
            catalog_federation_completeness=completeness,
        )
        return response, pipeline, observation

    def select_and_handoff(
        self,
        handoff_request: MachineCapabilityAcquisitionHandoffRequest,
    ) -> MachineCapabilityAcquisitionHandoffResponse:
        acquisition = handoff_request.acquisition_request
        selection = handoff_request.selection
        acquire_response, pipeline, observation = self._acquire_with_pipeline(
            acquisition,
            operation_discovery_correlation_id=selection.discovery_correlation_id,
        )
        if (
            selection.discovery_correlation_id
            != acquire_response.discovery_correlation_id
        ):
            raise MachineCapabilityAcquisitionSelectionError(
                "selection discovery_correlation_id must match acquisition response",
            )
        if (
            acquire_response.outcome
            is not MachineCapabilityAcquisitionOutcome.RECOMMENDATIONS_AVAILABLE
        ):
            raise MachineCapabilityAcquisitionSelectionError(
                "cannot handoff without governed recommendations from acquisition",
            )
        recommended_keys = {
            item.release.release_sort_key for item in acquire_response.recommendations
        }
        if selection.selected_release.release_sort_key not in recommended_keys:
            raise MachineCapabilityAcquisitionSelectionError(
                "selected release is not in governed recommendation set",
            )
        identity_key = CapabilityIdentityKey.from_discovery_identity(
            selection.selected_release.discovery,
        )
        consumer_target = selection.consumer_target
        if consumer_target is None:
            consumer_target = consumer_target_for_kind(
                selection.selected_release.discovery.kind,
            )
        try:
            delivery = (
                self.handoff_orchestrator.deliver_explicit_selection_from_pipeline(
                    pipeline=pipeline,
                    marketplace_query_context=acquisition.marketplace_query_context,
                    selected_identity_key=identity_key,
                    consumer_target=consumer_target,
                    selector_id=selection.selector_id,
                    discovery_correlation_id=selection.discovery_correlation_id,
                    selection_id=selection.selection_id,
                    handoff_id=handoff_request.handoff_id,
                    query_correlation_id=(
                        acquisition.observation.query_correlation_id
                        if acquisition.observation is not None
                        else None
                    ),
                    observation=observation,
                )
            )
        except MarketplaceHandoffSelectionError as exc:
            raise MachineCapabilityAcquisitionSelectionError(str(exc)) from exc
        return MachineCapabilityAcquisitionHandoffResponse(
            request_id=acquisition.request_id,
            discovery_correlation_id=selection.discovery_correlation_id,
            selection_id=selection.selection_id,
            handoff_id=handoff_request.handoff_id,
            delivery=delivery,
        )


__all__ = ["MachineCapabilityAcquisitionService"]
