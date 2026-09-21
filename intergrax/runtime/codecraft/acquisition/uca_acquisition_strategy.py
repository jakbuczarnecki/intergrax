# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UCA acquisition strategy adapter over public CodeCraft gap synthesis port."""

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
from intergrax.contracts.codecraft.gap_synthesis import (
    CodeCraftGapSynthesisOutcome,
    CodeCraftGapSynthesisPort,
    CodeCraftGapSynthesisRequest,
    CodeCraftGapSynthesisResult,
)

CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID = "codecraft.synthesis.v1"

_CODECRAFT_SYNTHESIS_KINDS = frozenset({CapabilityKind.TOOL})


class CodeCraftGapCapabilityAcquisitionStrategy:
    """Plugin strategy — maps UCA requests to CodeCraftGapSynthesisPort."""

    def __init__(self, port: CodeCraftGapSynthesisPort) -> None:
        self._port = port

    @property
    def strategy_id(self) -> str:
        return CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID

    @property
    def supported_kinds(self) -> frozenset[CapabilityKind]:
        return _CODECRAFT_SYNTHESIS_KINDS

    def supports(self, request: CapabilityAcquisitionRequest) -> bool:
        need = request.capability_need
        if need is None:
            need = CapabilityNeed(need_id=request.capability_gap.need_id)
        if not need.kinds:
            return True
        return bool(_CODECRAFT_SYNTHESIS_KINDS.intersection(need.kinds))

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
                reason_detail="capability_need required for codecraft synthesis",
            )
        if not need.intent_summary:
            return _terminal(
                request=request,
                outcome=CapabilityAcquisitionOutcome.FAILED,
                reason_code=CapabilityAcquisitionReasonCode.INVALID_REQUEST,
                started_at=started_at,
                reason_detail="capability_need.intent_summary required for synthesis goal",
            )

        target_kind = _target_kind_for_need(need)
        if target_kind not in _CODECRAFT_SYNTHESIS_KINDS:
            return _terminal(
                request=request,
                outcome=CapabilityAcquisitionOutcome.NOT_SUPPORTED,
                reason_code=CapabilityAcquisitionReasonCode.STRATEGY_REJECTED,
                started_at=started_at,
                reason_detail=f"codecraft synthesis does not support kind {target_kind.value}",
            )

        gap = request.capability_gap
        domain_request = CodeCraftGapSynthesisRequest(
            operation_id=request.request_id,
            gap_id=gap.gap_id,
            canonical_discovery_correlation_id=gap.discovery_correlation_id,
            capability_need=need,
            synthesis_goal=need.intent_summary,
            target_kind=target_kind,
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )
        domain_result = self._port.synthesize_from_gap(domain_request)
        integrity = _validate_domain_identity(
            request=request,
            domain_result=domain_result,
            started_at=started_at,
        )
        if integrity is not None:
            return integrity
        return _map_domain_result(
            request=request,
            domain_result=domain_result,
            started_at=started_at,
        )


def _target_kind_for_need(need: CapabilityNeed) -> CapabilityKind:
    if not need.kinds:
        return CapabilityKind.TOOL
    for kind in need.kinds:
        if kind in _CODECRAFT_SYNTHESIS_KINDS:
            return kind
    return need.kinds[0]


def _validate_domain_identity(
    *,
    request: CapabilityAcquisitionRequest,
    domain_result: CodeCraftGapSynthesisResult,
    started_at: datetime,
) -> CapabilityAcquisitionResult | None:
    if domain_result.operation_id != request.request_id:
        return _terminal(
            request=request,
            outcome=CapabilityAcquisitionOutcome.FAILED,
            reason_code=CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT,
            started_at=started_at,
            reason_detail="codecraft port returned mismatched operation_id",
        )
    gap_id = request.capability_gap.gap_id
    if domain_result.gap_id != gap_id:
        return _terminal(
            request=request,
            outcome=CapabilityAcquisitionOutcome.FAILED,
            reason_code=CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT,
            started_at=started_at,
            reason_detail="codecraft port returned mismatched gap_id",
        )
    if (
        request.correlation_id is not None
        and domain_result.correlation_id != request.correlation_id
    ):
        return _terminal(
            request=request,
            outcome=CapabilityAcquisitionOutcome.FAILED,
            reason_code=CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT,
            started_at=started_at,
            reason_detail="codecraft port returned mismatched correlation_id",
        )
    if (
        request.causation_id is not None
        and domain_result.causation_id != request.causation_id
    ):
        return _terminal(
            request=request,
            outcome=CapabilityAcquisitionOutcome.FAILED,
            reason_code=CapabilityAcquisitionReasonCode.EVIDENCE_INCONSISTENT,
            started_at=started_at,
            reason_detail="codecraft port returned mismatched causation_id",
        )
    return None


def _map_domain_result(
    *,
    request: CapabilityAcquisitionRequest,
    domain_result: CodeCraftGapSynthesisResult,
    started_at: datetime,
) -> CapabilityAcquisitionResult:
    outcome = domain_result.outcome
    completed_at = datetime.now(tz=UTC)
    if outcome is CodeCraftGapSynthesisOutcome.SUCCEEDED:
        evidence = CapabilityAcquisitionEvidence(
            domain_handoff_reference=domain_result.domain_handoff_reference,
            artifact_reference=domain_result.artifact_reference,
            evidence_ref=domain_result.codecraft_operation_correlation_id,
        )
        return CapabilityAcquisitionResult(
            request_id=request.request_id,
            gap_id=request.capability_gap.gap_id,
            strategy_id=CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID,
            outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
            reason_code=CapabilityAcquisitionReasonCode.NONE,
            started_at=started_at,
            completed_at=completed_at,
            evidence=evidence,
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )
    if outcome is CodeCraftGapSynthesisOutcome.NOT_SUPPORTED:
        return _terminal(
            request=request,
            outcome=CapabilityAcquisitionOutcome.NOT_SUPPORTED,
            reason_code=CapabilityAcquisitionReasonCode.STRATEGY_REJECTED,
            started_at=started_at,
            reason_detail=domain_result.reason_detail
            or "codecraft synthesis not supported",
        )
    if outcome is CodeCraftGapSynthesisOutcome.UNAVAILABLE:
        return _terminal(
            request=request,
            outcome=CapabilityAcquisitionOutcome.UNAVAILABLE,
            reason_code=CapabilityAcquisitionReasonCode.STRATEGY_UNAVAILABLE,
            started_at=started_at,
            reason_detail=domain_result.reason_detail,
        )
    if outcome is CodeCraftGapSynthesisOutcome.BLOCKED:
        return _terminal(
            request=request,
            outcome=CapabilityAcquisitionOutcome.BLOCKED,
            reason_code=CapabilityAcquisitionReasonCode.DOMAIN_HANDOFF_REJECTED,
            started_at=started_at,
            reason_detail=domain_result.reason_detail,
        )
    if outcome is CodeCraftGapSynthesisOutcome.REQUIRES_HITL:
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
        strategy_id=CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID,
        outcome=outcome,
        reason_code=reason_code,
        started_at=started_at,
        completed_at=datetime.now(tz=UTC),
        reason_detail=reason_detail,
        correlation_id=request.correlation_id,
        causation_id=request.causation_id,
    )


__all__ = [
    "CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID",
    "CodeCraftGapCapabilityAcquisitionStrategy",
]
