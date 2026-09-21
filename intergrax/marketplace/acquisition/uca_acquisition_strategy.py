# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UCA acquisition strategy adapter over public Marketplace gap acquisition port."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.need import CapabilityNeed
from intergrax.contracts.capability_catalog.query import CapabilityDiscoveryQuery
from intergrax.contracts.capability_catalog.scope import (
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
)
from intergrax.contracts.marketplace.gap_acquisition import (
    MarketplaceGapAcquisitionOutcome,
    MarketplaceGapAcquisitionPort,
    MarketplaceGapAcquisitionRequest,
)
from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext

MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID = "marketplace.gap_acquisition.v1"

_MARKETPLACE_ACQUISITION_KINDS = frozenset(
    {CapabilityKind.TOOL, CapabilityKind.SKILL, CapabilityKind.AGENT},
)


class MarketplaceGapCapabilityAcquisitionStrategy:
    """Plugin strategy — maps UCA requests to MarketplaceGapAcquisitionPort."""

    def __init__(
        self,
        port: MarketplaceGapAcquisitionPort,
        *,
        marketplace_query_context: MarketplaceQueryContext | None = None,
    ) -> None:
        self._port = port
        self._marketplace_query_context = (
            marketplace_query_context or MarketplaceQueryContext()
        )

    @property
    def strategy_id(self) -> str:
        return MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID

    @property
    def supported_kinds(self) -> frozenset[CapabilityKind]:
        return _MARKETPLACE_ACQUISITION_KINDS

    def supports(self, request: CapabilityAcquisitionRequest) -> bool:
        need = request.capability_need
        if need is None:
            need = CapabilityNeed(need_id=request.capability_gap.need_id)
        if not need.kinds:
            return True
        return bool(_MARKETPLACE_ACQUISITION_KINDS.intersection(need.kinds))

    def acquire(
        self, request: CapabilityAcquisitionRequest
    ) -> CapabilityAcquisitionResult:
        started_at = datetime.now(tz=UTC)
        need = request.capability_need
        if need is None:
            return _terminal(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.INVALID_REQUEST,
                started_at=started_at,
                reason_detail="capability_need required for marketplace acquisition",
            )

        discovery_query = _discovery_query_for_need(need)
        gap = request.capability_gap
        domain_request = MarketplaceGapAcquisitionRequest(
            operation_id=request.request_id,
            gap_id=gap.gap_id,
            canonical_discovery_correlation_id=gap.discovery_correlation_id,
            capability_need=need,
            discovery_query=discovery_query,
            marketplace_query_context=self._marketplace_query_context,
            query_text=need.intent_summary,
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )
        domain_result = self._port.acquire_from_gap(domain_request)
        if domain_result.operation_id != request.request_id:
            return _terminal(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                reason_detail="marketplace port returned mismatched operation_id",
            )
        if domain_result.gap_id != gap.gap_id:
            return _terminal(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT,
                started_at=started_at,
                reason_detail="marketplace port returned mismatched gap_id",
            )
        return _map_domain_result(
            request=request,
            domain_result=domain_result,
            started_at=started_at,
        )


def _discovery_query_for_need(need: CapabilityNeed) -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
        kinds=need.kinds,
    )


def _map_domain_result(
    *,
    request: CapabilityAcquisitionRequest,
    domain_result,
    started_at: datetime,
) -> CapabilityAcquisitionResult:
    outcome = domain_result.outcome
    completed_at = datetime.now(tz=UTC)
    if outcome is MarketplaceGapAcquisitionOutcome.SUCCEEDED:
        evidence = CapabilityAcquisitionEvidence(
            domain_handoff_reference=domain_result.domain_handoff_reference,
            artifact_reference=domain_result.artifact_reference,
            evidence_ref=domain_result.marketplace_listing_correlation_id,
        )
        return CapabilityAcquisitionResult(
            request_id=request.request_id,
            gap_id=request.capability_gap.gap_id,
            strategy_id=MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID,
            outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
            reason_code=CapabilityAcquisitionReasonCode.NONE,
            started_at=started_at,
            completed_at=completed_at,
            evidence=evidence,
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )
    if outcome is MarketplaceGapAcquisitionOutcome.NO_ACQUISITION_SOURCE:
        return _terminal(
            request=request,
            outcome=CapabilityAcquisitionOutcome.NOT_SUPPORTED,
            reason_code=CapabilityAcquisitionReasonCode.STRATEGY_REJECTED,
            started_at=started_at,
            reason_detail=domain_result.reason_detail
            or "no marketplace acquisition source",
        )
    if outcome is MarketplaceGapAcquisitionOutcome.UNAVAILABLE:
        return _terminal(
            request=request,
            outcome=CapabilityAcquisitionOutcome.UNAVAILABLE,
            reason_code=CapabilityAcquisitionReasonCode.STRATEGY_UNAVAILABLE,
            started_at=started_at,
            reason_detail=domain_result.reason_detail,
        )
    if outcome is MarketplaceGapAcquisitionOutcome.BLOCKED:
        return _terminal(
            request=request,
            outcome=CapabilityAcquisitionOutcome.BLOCKED,
            reason_code=CapabilityAcquisitionReasonCode.DOMAIN_HANDOFF_REJECTED,
            started_at=started_at,
            reason_detail=domain_result.reason_detail,
        )
    if outcome is MarketplaceGapAcquisitionOutcome.REQUIRES_HITL:
        return _terminal(
            request=request,
            outcome=CapabilityAcquisitionOutcome.REQUIRES_HITL,
            reason_code=CapabilityAcquisitionReasonCode.HUMAN_APPROVAL_REQUIRED,
            started_at=started_at,
            reason_detail=domain_result.reason_detail,
        )
    return _terminal(
        request=request,
        outcome=CapabilityAcquisitionOutcome.FAILED,
        reason_code=CapabilityAcquisitionReasonCode.STRATEGY_FAILED,
        started_at=started_at,
        reason_detail=domain_result.reason_detail,
    )


def _terminal(
    *,
    request: CapabilityAcquisitionRequest,
    outcome: CapabilityAcquisitionOutcome,
    reason_code: CapabilityAcquisitionReasonCode,
    started_at: datetime,
    reason_detail: str = "",
) -> CapabilityAcquisitionResult:
    return CapabilityAcquisitionResult(
        request_id=request.request_id,
        gap_id=request.capability_gap.gap_id,
        strategy_id=MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID,
        outcome=outcome,
        reason_code=reason_code,
        started_at=started_at,
        completed_at=datetime.now(tz=UTC),
        reason_detail=reason_detail,
        correlation_id=request.correlation_id,
        causation_id=request.causation_id,
    )


__all__ = [
    "MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID",
    "MarketplaceGapCapabilityAcquisitionStrategy",
]
