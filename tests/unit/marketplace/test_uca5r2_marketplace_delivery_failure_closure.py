# © Artur Czarnecki. All rights reserved.

"""UCA-5R2 — close expected marketplace handoff delivery failures into typed results."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.capability_acquisition.acquisition_service import (
    CapabilityAcquisitionService,
)
from intergrax.capability_acquisition.permit_acquisition_authorization import (
    PermitCapabilityAcquisitionAuthorizationPort,
)
from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    FederatedCapabilityCatalog,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
    derive_capability_acquisition_request_id,
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
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.capability_catalog.discovery_completion import (
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.marketplace import (
    MarketplaceListingRecord,
    MarketplaceQueryContext,
)
from intergrax.contracts.marketplace.gap_acquisition import (
    MarketplaceGapAcquisitionOutcome,
    MarketplaceGapAcquisitionRequest,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumer,
    CapabilityHandoffConsumerError,
    CapabilityHandoffDeliveryAdmissionError,
    CapabilityHandoffDeliveryAdmissionResult,
    CapabilityHandoffEnvelope,
    CapabilityHandoffIdentityConflictError,
)
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
    MarketplaceRecommendationService,
)
from intergrax.marketplace.acquisition import (
    MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID,
    MarketplaceGapAcquisitionService,
    MarketplaceGapCapabilityAcquisitionStrategy,
)
from intergrax.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryService,
    InMemoryCapabilityHandoffDeliveryAdmission,
    InMemoryCapabilityHandoffTraceEvidenceConsumer,
    MarketplaceDiscoveryHandoffOrchestrator,
)

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.intergrax.uca5r2",
    source_kind=CapabilitySourceKind.OFFICIAL,
)
_CREATED = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)


def _discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


def _gap_request(
    *, operation_id: str = "op-uca5r2"
) -> MarketplaceGapAcquisitionRequest:
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
        correlation_id="corr-r2",
        causation_id="cause-r2",
    )


class _OkConsumer:
    @property
    def consumer_id(self) -> str:
        return "uca5r2.test.consumer"

    def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
        return


def _gap_service(
    *,
    admission,
    consumer: CapabilityHandoffConsumer | None = None,
) -> MarketplaceGapAcquisitionService:
    source = MarketplaceCapabilityCatalogSource(
        source=_OFFICIAL,
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.TOOL,
                logical_id="tool.uca5r2",
                display_label="tool.uca5r2",
                publisher="intergrax",
            ),
        ),
    )
    catalog = MarketplaceCatalogService(
        catalog=FederatedCapabilityCatalog((source,)),
        marketplace_sources=(source,),
    )
    delivery = CapabilityHandoffDeliveryService(
        consumer=consumer or _OkConsumer(),
        delivery_admission=admission,
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


class _ReserveIdentityConflictAdmission:
    def reserve(
        self, envelope: CapabilityHandoffEnvelope
    ) -> CapabilityHandoffDeliveryAdmissionResult:
        raise CapabilityHandoffIdentityConflictError(
            "same handoff_id with different envelope payload",
        )

    def mark_delivered(self, handoff_id: str) -> None:
        return

    def mark_delivery_failed(self, handoff_id: str) -> None:
        return


def test_r1_identity_conflict_returns_typed_result() -> None:
    service = _gap_service(admission=_ReserveIdentityConflictAdmission())
    result = service.acquire_from_gap(_gap_request())
    assert result.outcome is MarketplaceGapAcquisitionOutcome.FAILED
    assert result.domain_handoff_reference is None
    assert result.artifact_reference is None


class _ReserveAdmissionErrorAdmission:
    def reserve(
        self, envelope: CapabilityHandoffEnvelope
    ) -> CapabilityHandoffDeliveryAdmissionResult:
        raise CapabilityHandoffDeliveryAdmissionError("admission store unavailable")

    def mark_delivered(self, handoff_id: str) -> None:
        return

    def mark_delivery_failed(self, handoff_id: str) -> None:
        return


def test_r2_admission_error_returns_typed_result() -> None:
    service = _gap_service(admission=_ReserveAdmissionErrorAdmission())
    result = service.acquire_from_gap(_gap_request())
    assert result.outcome is MarketplaceGapAcquisitionOutcome.UNAVAILABLE


class _FailingReleaseAdmission:
    def __init__(self) -> None:
        self._inner = InMemoryCapabilityHandoffDeliveryAdmission()

    def reserve(
        self, envelope: CapabilityHandoffEnvelope
    ) -> CapabilityHandoffDeliveryAdmissionResult:
        return self._inner.reserve(envelope)

    def mark_delivered(self, handoff_id: str) -> None:
        self._inner.mark_delivered(handoff_id)

    def mark_delivery_failed(self, handoff_id: str) -> None:
        raise RuntimeError("release store down")


class _FailingConsumer:
    @property
    def consumer_id(self) -> str:
        return "uca5r2.test.consumer"

    def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
        raise CapabilityHandoffConsumerError("consumer rejected")


def test_r3_lifecycle_transition_error_returns_typed_failed() -> None:
    service = _gap_service(
        admission=_FailingReleaseAdmission(),
        consumer=_FailingConsumer(),
    )
    result = service.acquire_from_gap(_gap_request())
    assert result.outcome is MarketplaceGapAcquisitionOutcome.FAILED


class _FailingMarkDeliveredAdmission:
    def __init__(self) -> None:
        self._inner = InMemoryCapabilityHandoffDeliveryAdmission()

    def reserve(
        self, envelope: CapabilityHandoffEnvelope
    ) -> CapabilityHandoffDeliveryAdmissionResult:
        return self._inner.reserve(envelope)

    def mark_delivered(self, handoff_id: str) -> None:
        raise CapabilityHandoffDeliveryAdmissionError("commit failed")

    def mark_delivery_failed(self, handoff_id: str) -> None:
        self._inner.mark_delivery_failed(handoff_id)


def test_r4_outcome_uncertain_never_succeeded() -> None:
    service = _gap_service(admission=_FailingMarkDeliveredAdmission())
    result = service.acquire_from_gap(_gap_request())
    assert result.outcome is MarketplaceGapAcquisitionOutcome.FAILED
    assert result.outcome is not MarketplaceGapAcquisitionOutcome.SUCCEEDED
    assert "delivery lifecycle could not be committed" in result.reason_detail


@pytest.mark.parametrize(
    "admission",
    [
        _ReserveIdentityConflictAdmission(),
        _ReserveAdmissionErrorAdmission(),
        _FailingReleaseAdmission(),
        _FailingMarkDeliveredAdmission(),
    ],
)
def test_expected_delivery_errors_do_not_escape_port(admission) -> None:
    consumer = (
        _FailingConsumer()
        if isinstance(admission, _FailingReleaseAdmission)
        else _OkConsumer()
    )
    service = _gap_service(admission=admission, consumer=consumer)
    result = service.acquire_from_gap(_gap_request(operation_id="op-no-escape"))
    assert isinstance(result.outcome, MarketplaceGapAcquisitionOutcome)


def _uca_gap() -> CapabilityGap:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="canonical-corr",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_CREATED,
    )
    return CapabilityGap.from_discovery_completion(completion)


def test_uca_e2e_delivery_admission_failure_closure() -> None:
    gap = _uca_gap()
    need = CapabilityNeed(
        need_id=gap.need_id,
        kinds=(CapabilityKind.TOOL,),
        intent_summary="tool",
    )
    request = CapabilityAcquisitionRequest(
        request_id=derive_capability_acquisition_request_id(
            gap_id=gap.gap_id,
            request_nonce="nonce-r2",
        ),
        request_nonce="nonce-r2",
        capability_gap=gap,
        capability_need=need,
        correlation_id="corr-e2e",
        causation_id="cause-e2e",
        requested_at=_CREATED,
    )
    gap_service = _gap_service(admission=_ReserveAdmissionErrorAdmission())
    strategy = MarketplaceGapCapabilityAcquisitionStrategy(gap_service)
    service = CapabilityAcquisitionService(
        (strategy,),
        authorization=PermitCapabilityAcquisitionAuthorizationPort(),
    )
    result = service.acquire(request)
    assert result.outcome is CapabilityAcquisitionOutcome.UNAVAILABLE
    assert result.reason_code is CapabilityAcquisitionReasonCode.STRATEGY_UNAVAILABLE
    assert result.strategy_id == MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID
    assert result.request_id == request.request_id
    assert result.gap_id == gap.gap_id
    assert result.correlation_id == "corr-e2e"
    assert result.causation_id == "cause-e2e"
    assert result.evidence is None
