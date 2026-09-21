# © Artur Czarnecki. All rights reserved.

"""UCA-5R — marketplace gap success contract and typed handoff failure mapping."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    FederatedCapabilityCatalog,
)
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernanceContext,
    CapabilityKind,
    CapabilityNeed,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace import (
    MarketplaceListingRecord,
    MarketplaceQueryContext,
)
from intergrax.contracts.marketplace.gap_acquisition import (
    MarketplaceGapAcquisitionOutcome,
    MarketplaceGapAcquisitionRequest,
    MarketplaceGapAcquisitionResult,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerError,
    CapabilityHandoffConsumerFailureDisposition,
)
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
    MarketplaceRecommendationService,
)
from intergrax.marketplace.acquisition import MarketplaceGapAcquisitionService
from intergrax.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryService,
    InMemoryCapabilityHandoffDeliveryAdmission,
    InMemoryCapabilityHandoffTraceEvidenceConsumer,
    MarketplaceDiscoveryHandoffOrchestrator,
)

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.intergrax.uca5r",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


def _gap_request(*, operation_id: str = "op-uca5r") -> MarketplaceGapAcquisitionRequest:
    return MarketplaceGapAcquisitionRequest(
        operation_id=operation_id,
        gap_id="capability-gap:need-1:canonical-corr",
        canonical_discovery_correlation_id="canonical-corr",
        capability_need=CapabilityNeed(
            need_id="need-1",
            kinds=(CapabilityKind.TOOL,),
            intent_summary="tool",
        ),
        discovery_query=_discovery_query(),
        marketplace_query_context=MarketplaceQueryContext(),
    )


def test_r1_subjectless_succeeded_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        MarketplaceGapAcquisitionResult(
            operation_id="x",
            gap_id="g",
            outcome=MarketplaceGapAcquisitionOutcome.SUCCEEDED,
        )


def test_r1_succeeded_with_domain_handoff_reference_is_valid() -> None:
    result = MarketplaceGapAcquisitionResult(
        operation_id="x",
        gap_id="g",
        outcome=MarketplaceGapAcquisitionOutcome.SUCCEEDED,
        domain_handoff_reference="handoff://x",
    )
    assert result.domain_handoff_reference == "handoff://x"


def test_r1_succeeded_with_artifact_reference_is_valid() -> None:
    result = MarketplaceGapAcquisitionResult(
        operation_id="x",
        gap_id="g",
        outcome=MarketplaceGapAcquisitionOutcome.SUCCEEDED,
        artifact_reference="artifact://x",
    )
    assert result.artifact_reference == "artifact://x"


class _DispositionConsumer:
    def __init__(
        self,
        *,
        disposition: CapabilityHandoffConsumerFailureDisposition,
        message: str,
    ) -> None:
        self._disposition = disposition
        self._message = message

    @property
    def consumer_id(self) -> str:
        return "uca5r.test.consumer"

    def consume(self, envelope) -> None:
        raise CapabilityHandoffConsumerError(
            self._message,
            disposition=self._disposition,
        )


def _gap_service_with_consumer(
    consumer: _DispositionConsumer,
) -> MarketplaceGapAcquisitionService:
    source = MarketplaceCapabilityCatalogSource(
        source=_OFFICIAL,
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.TOOL,
                logical_id="tool.uca5r",
                display_label="tool.uca5r",
                publisher="intergrax",
            ),
        ),
    )
    catalog = MarketplaceCatalogService(
        catalog=FederatedCapabilityCatalog((source,)),
        marketplace_sources=(source,),
    )
    delivery = CapabilityHandoffDeliveryService(
        consumer=consumer,
        delivery_admission=InMemoryCapabilityHandoffDeliveryAdmission(),
        trace_evidence_consumer=InMemoryCapabilityHandoffTraceEvidenceConsumer(),
    )
    orchestrator = MarketplaceDiscoveryHandoffOrchestrator(
        catalog_service=catalog,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(),
        delivery_service=delivery,
    )
    return MarketplaceGapAcquisitionService(
        catalog_service=catalog,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(),
        recommendation_service=MarketplaceRecommendationService.with_defaults(),
        handoff_orchestrator=orchestrator,
    )


@pytest.mark.parametrize(
    ("disposition", "expected"),
    [
        (
            CapabilityHandoffConsumerFailureDisposition.BLOCKED,
            MarketplaceGapAcquisitionOutcome.BLOCKED,
        ),
        (
            CapabilityHandoffConsumerFailureDisposition.UNAVAILABLE,
            MarketplaceGapAcquisitionOutcome.UNAVAILABLE,
        ),
        (
            CapabilityHandoffConsumerFailureDisposition.REQUIRES_HITL,
            MarketplaceGapAcquisitionOutcome.REQUIRES_HITL,
        ),
        (
            CapabilityHandoffConsumerFailureDisposition.FAILED,
            MarketplaceGapAcquisitionOutcome.FAILED,
        ),
    ],
)
def test_r2_typed_consumer_failure_maps_to_marketplace_outcome(
    disposition: CapabilityHandoffConsumerFailureDisposition,
    expected: MarketplaceGapAcquisitionOutcome,
) -> None:
    service = _gap_service_with_consumer(
        _DispositionConsumer(disposition=disposition, message="anything"),
    )
    result = service.acquire_from_gap(
        _gap_request(operation_id=f"op-{disposition.value}")
    )
    assert result.outcome is expected


def test_r2_adversarial_message_does_not_change_unavailable_outcome() -> None:
    service = _gap_service_with_consumer(
        _DispositionConsumer(
            disposition=CapabilityHandoffConsumerFailureDisposition.UNAVAILABLE,
            message="human approval block hitl reject",
        ),
    )
    result = service.acquire_from_gap(_gap_request(operation_id="op-adversarial"))
    assert result.outcome is MarketplaceGapAcquisitionOutcome.UNAVAILABLE


class _LegacyMessageConsumer:
    @property
    def consumer_id(self) -> str:
        return "uca5r.test.consumer"

    def consume(self, envelope) -> None:
        raise CapabilityHandoffConsumerError("approval required")


def test_r2_legacy_generic_consumer_error_maps_to_failed() -> None:
    legacy = CapabilityHandoffConsumerError("approval required")
    assert legacy.disposition is CapabilityHandoffConsumerFailureDisposition.FAILED
    service = _gap_service_with_consumer(_LegacyMessageConsumer())
    result = service.acquire_from_gap(_gap_request(operation_id="op-legacy"))
    assert result.outcome is MarketplaceGapAcquisitionOutcome.FAILED


def test_r2_same_disposition_different_messages_same_outcome() -> None:
    service_a = _gap_service_with_consumer(
        _DispositionConsumer(
            disposition=CapabilityHandoffConsumerFailureDisposition.BLOCKED,
            message="message-a",
        ),
    )
    service_b = _gap_service_with_consumer(
        _DispositionConsumer(
            disposition=CapabilityHandoffConsumerFailureDisposition.BLOCKED,
            message="message-b",
        ),
    )
    outcome_a = service_a.acquire_from_gap(_gap_request(operation_id="op-a")).outcome
    outcome_b = service_b.acquire_from_gap(_gap_request(operation_id="op-b")).outcome
    assert outcome_a is outcome_b is MarketplaceGapAcquisitionOutcome.BLOCKED
