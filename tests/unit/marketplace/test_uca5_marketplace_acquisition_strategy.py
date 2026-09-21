# © Artur Czarnecki. All rights reserved.

"""UCA-5 — marketplace gap acquisition strategy and seam."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.capability_acquisition.acquisition_service import (
    CapabilityAcquisitionService,
)
from intergrax.capability_acquisition.permit_acquisition_authorization import (
    PermitCapabilityAcquisitionAuthorizationPort,
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
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.capability_catalog.discovery_completion import (
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.need import CapabilityNeed
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
    derive_capability_qualification_request_id,
)
from intergrax.contracts.marketplace.gap_acquisition import (
    MarketplaceGapAcquisitionOutcome,
    MarketplaceGapAcquisitionPort,
    MarketplaceGapAcquisitionRequest,
    MarketplaceGapAcquisitionResult,
)
from intergrax.marketplace.acquisition import (
    MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID,
    MarketplaceGapCapabilityAcquisitionStrategy,
)

pytestmark = pytest.mark.unit

_CREATED = datetime(2026, 9, 21, 8, 0, tzinfo=UTC)


def _gap() -> CapabilityGap:
    completion = build_discovery_completion(
        need_id="need-uca5",
        discovery_correlation_id="canonical-discovery-corr",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_CREATED,
    )
    return CapabilityGap.from_discovery_completion(completion)


def _request(
    gap: CapabilityGap,
    *,
    kinds: tuple[CapabilityKind, ...] = (CapabilityKind.TOOL,),
) -> CapabilityAcquisitionRequest:
    need = CapabilityNeed(
        need_id=gap.need_id,
        kinds=kinds,
        intent_summary="acquire tool from marketplace",
    )
    return CapabilityAcquisitionRequest(
        request_id=derive_capability_acquisition_request_id(
            gap_id=gap.gap_id,
            request_nonce="nonce-uca5",
        ),
        request_nonce="nonce-uca5",
        capability_gap=gap,
        capability_need=need,
        correlation_id="corr-uca5",
        causation_id="cause-uca5",
        requested_at=_CREATED,
    )


class _RecordingGapPort(MarketplaceGapAcquisitionPort):
    def __init__(self, result: MarketplaceGapAcquisitionResult) -> None:
        self.result = result
        self.calls: list[MarketplaceGapAcquisitionRequest] = []

    def acquire_from_gap(
        self,
        request: MarketplaceGapAcquisitionRequest,
    ) -> MarketplaceGapAcquisitionResult:
        self.calls.append(request)
        return self.result


class _UnavailablePort(MarketplaceGapAcquisitionPort):
    def acquire_from_gap(
        self,
        request: MarketplaceGapAcquisitionRequest,
    ) -> MarketplaceGapAcquisitionResult:
        return MarketplaceGapAcquisitionResult(
            operation_id=request.operation_id,
            gap_id=request.gap_id,
            outcome=MarketplaceGapAcquisitionOutcome.UNAVAILABLE,
            reason_detail="marketplace down",
        )


def test_strategy_supports_is_side_effect_free() -> None:
    port = _RecordingGapPort(
        MarketplaceGapAcquisitionResult(
            operation_id="operation-id",
            gap_id="gap-id",
            outcome=MarketplaceGapAcquisitionOutcome.FAILED,
        ),
    )
    strategy = MarketplaceGapCapabilityAcquisitionStrategy(port)
    req = _request(_gap())
    assert strategy.supports(req) is True
    assert port.calls == []


def test_strategy_does_not_invoke_machine_acquire_api() -> None:
    gap = _gap()
    req = _request(gap)
    port = _RecordingGapPort(
        MarketplaceGapAcquisitionResult(
            operation_id=req.request_id,
            gap_id=gap.gap_id,
            outcome=MarketplaceGapAcquisitionOutcome.SUCCEEDED,
            domain_handoff_reference="handoff://h-1",
            marketplace_listing_correlation_id="marketplace-gap-listing:op",
        ),
    )
    strategy = MarketplaceGapCapabilityAcquisitionStrategy(port)
    result = strategy.acquire(req)
    assert result.outcome is CapabilityAcquisitionOutcome.SUCCEEDED
    assert result.evidence is not None
    assert result.evidence.domain_handoff_reference == "handoff://h-1"
    assert len(port.calls) == 1
    mapped = port.calls[0]
    assert mapped.operation_id == req.request_id
    assert mapped.gap_id == gap.gap_id
    assert mapped.canonical_discovery_correlation_id == gap.discovery_correlation_id
    assert mapped.correlation_id == req.correlation_id
    assert mapped.causation_id == req.causation_id


def test_offer_without_handoff_is_not_succeeded() -> None:
    gap = _gap()
    req = _request(gap)
    port = _RecordingGapPort(
        MarketplaceGapAcquisitionResult(
            operation_id=req.request_id,
            gap_id=gap.gap_id,
            outcome=MarketplaceGapAcquisitionOutcome.NO_ACQUISITION_SOURCE,
            reason_detail="listing only",
        ),
    )
    strategy = MarketplaceGapCapabilityAcquisitionStrategy(port)
    result = strategy.acquire(req)
    assert result.outcome is CapabilityAcquisitionOutcome.NOT_SUPPORTED
    assert result.reason_code is CapabilityAcquisitionReasonCode.STRATEGY_REJECTED


def test_marketplace_unavailable_maps_to_unavailable_not_no_strategy() -> None:
    gap = _gap()
    req = _request(gap)
    strategy = MarketplaceGapCapabilityAcquisitionStrategy(_UnavailablePort())
    service = CapabilityAcquisitionService(
        (strategy,),
        authorization=PermitCapabilityAcquisitionAuthorizationPort(),
    )
    result = service.acquire(req)
    assert result.outcome is CapabilityAcquisitionOutcome.UNAVAILABLE
    assert result.strategy_id == MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID
    assert result.outcome is not CapabilityAcquisitionOutcome.NO_STRATEGY


def test_wrong_port_identity_fails_integrity() -> None:
    gap = _gap()
    req = _request(gap)
    port = _RecordingGapPort(
        MarketplaceGapAcquisitionResult(
            operation_id="wrong-id",
            gap_id=gap.gap_id,
            outcome=MarketplaceGapAcquisitionOutcome.SUCCEEDED,
            domain_handoff_reference="handoff://x",
        ),
    )
    strategy = MarketplaceGapCapabilityAcquisitionStrategy(port)
    result = strategy.acquire(req)
    assert result.outcome is CapabilityAcquisitionOutcome.FAILED
    assert result.reason_code is CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT


def test_succeeded_result_is_qualification_request_compatible() -> None:
    gap = _gap()
    req = _request(gap)
    port = _RecordingGapPort(
        MarketplaceGapAcquisitionResult(
            operation_id=req.request_id,
            gap_id=gap.gap_id,
            outcome=MarketplaceGapAcquisitionOutcome.SUCCEEDED,
            domain_handoff_reference="handoff://gap-handoff",
        ),
    )
    strategy = MarketplaceGapCapabilityAcquisitionStrategy(port)
    acquisition = strategy.acquire(req)
    qual = CapabilityQualificationRequest(
        qualification_request_id=derive_capability_qualification_request_id(
            acquisition_request_id=req.request_id,
            qualification_nonce="q-nonce",
        ),
        qualification_nonce="q-nonce",
        acquisition_request_id=req.request_id,
        gap_id=gap.gap_id,
        strategy_id=MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID,
        acquisition_result=acquisition,
        correlation_id=req.correlation_id,
        causation_id=req.causation_id,
        requested_at=_CREATED,
    )
    assert qual.acquisition_result.outcome is CapabilityAcquisitionOutcome.SUCCEEDED


def test_blocked_and_hitl_mapping() -> None:
    gap = _gap()
    req = _request(gap)
    blocked = MarketplaceGapCapabilityAcquisitionStrategy(
        _RecordingGapPort(
            MarketplaceGapAcquisitionResult(
                operation_id=req.request_id,
                gap_id=gap.gap_id,
                outcome=MarketplaceGapAcquisitionOutcome.BLOCKED,
            ),
        ),
    )
    assert blocked.acquire(req).outcome is CapabilityAcquisitionOutcome.BLOCKED
    hitl = MarketplaceGapCapabilityAcquisitionStrategy(
        _RecordingGapPort(
            MarketplaceGapAcquisitionResult(
                operation_id=req.request_id,
                gap_id=gap.gap_id,
                outcome=MarketplaceGapAcquisitionOutcome.REQUIRES_HITL,
            ),
        ),
    )
    assert hitl.acquire(req).outcome is CapabilityAcquisitionOutcome.REQUIRES_HITL
