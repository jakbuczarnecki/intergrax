# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool domain capability realization provider adapter (UCA-2)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.capability_acquisition.evidence import (
    CapabilityRealizationEvidence,
)
from intergrax.contracts.capability_acquisition.outcome import (
    CapabilityRealizationOutcome,
)
from intergrax.contracts.capability_acquisition.reason_code import (
    CapabilityRealizationReasonCode,
)
from intergrax.contracts.capability_acquisition.request import (
    CapabilityRealizationRequest,
)
from intergrax.contracts.capability_acquisition.result import (
    CapabilityRealizationResult,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.tools.known_capability_realization import (
    KnownToolCapabilityRealizationOutcome,
    KnownToolCapabilityRealizationPort,
    KnownToolCapabilityRealizationRequest,
)

TOOL_CAPABILITY_REALIZATION_PROVIDER_ID = "tool.known_capability_realization.v1"


class ToolCapabilityRealizationProvider:
    """Maps generic realization SPI to Tool public handoff contract."""

    def __init__(self, handoff: KnownToolCapabilityRealizationPort) -> None:
        self._handoff = handoff

    @property
    def provider_id(self) -> str:
        return TOOL_CAPABILITY_REALIZATION_PROVIDER_ID

    @property
    def supported_kinds(self) -> frozenset[CapabilityKind]:
        return frozenset({CapabilityKind.TOOL})

    def supports(self, request: CapabilityRealizationRequest) -> bool:
        return request.capability_kind is CapabilityKind.TOOL

    def realize(self, request: CapabilityRealizationRequest) -> CapabilityRealizationResult:
        started_at = datetime.now(tz=UTC)
        host_profile_id = request.host_profile_id
        if host_profile_id is None:
            return _failure(
                request=request,
                outcome=CapabilityRealizationOutcome.FAILED,
                reason_code=CapabilityRealizationReasonCode.INVALID_REQUEST,
                started_at=started_at,
                reason_detail="host_profile_id required for tool realization",
            )

        domain_request = KnownToolCapabilityRealizationRequest(
            operation_id=request.request_id,
            host_profile_id=host_profile_id,
            capability_identity=request.realization_need.capability_identity,
            requested_at=request.requested_at,
        )
        domain_result = self._handoff.realize(domain_request)
        completed_at = datetime.now(tz=UTC)
        mapped = _map_domain_outcome(domain_result.outcome)
        reason_code = _map_domain_reason(domain_result.outcome)
        evidence: CapabilityRealizationEvidence | None = None
        if mapped is CapabilityRealizationOutcome.SUCCEEDED:
            if domain_result.availability_evidence is None:
                return _failure(
                    request=request,
                    outcome=CapabilityRealizationOutcome.FAILED,
                    reason_code=CapabilityRealizationReasonCode.EVIDENCE_INCONSISTENT,
                    started_at=started_at,
                    reason_detail="tool domain succeeded without availability evidence",
                )
            evidence = CapabilityRealizationEvidence.from_availability_evidence(
                domain_result.availability_evidence,
                domain_reference=domain_result.domain_reference,
            )

        return CapabilityRealizationResult(
            request_id=request.request_id,
            realization_need_id=request.realization_need.realization_need_id,
            provider_id=self.provider_id,
            outcome=mapped,
            reason_code=reason_code,
            capability_identity=request.realization_need.capability_identity,
            started_at=started_at,
            completed_at=completed_at,
            evidence=evidence,
            reason_detail=domain_result.reason_detail,
        )


def _map_domain_outcome(
    outcome: KnownToolCapabilityRealizationOutcome,
) -> CapabilityRealizationOutcome:
    if outcome in (
        KnownToolCapabilityRealizationOutcome.REALIZED,
        KnownToolCapabilityRealizationOutcome.ALREADY_REALIZED,
    ):
        return CapabilityRealizationOutcome.SUCCEEDED
    if outcome is KnownToolCapabilityRealizationOutcome.BLOCKED:
        return CapabilityRealizationOutcome.BLOCKED
    if outcome is KnownToolCapabilityRealizationOutcome.UNAVAILABLE:
        return CapabilityRealizationOutcome.UNAVAILABLE
    if outcome is KnownToolCapabilityRealizationOutcome.REQUIRES_HITL:
        return CapabilityRealizationOutcome.REQUIRES_HITL
    return CapabilityRealizationOutcome.FAILED


def _map_domain_reason(
    outcome: KnownToolCapabilityRealizationOutcome,
) -> CapabilityRealizationReasonCode:
    if outcome in (
        KnownToolCapabilityRealizationOutcome.REALIZED,
        KnownToolCapabilityRealizationOutcome.ALREADY_REALIZED,
    ):
        return CapabilityRealizationReasonCode.NONE
    if outcome is KnownToolCapabilityRealizationOutcome.BLOCKED:
        return CapabilityRealizationReasonCode.DOMAIN_HANDOFF_REJECTED
    if outcome is KnownToolCapabilityRealizationOutcome.UNAVAILABLE:
        return CapabilityRealizationReasonCode.PROVIDER_UNAVAILABLE
    if outcome is KnownToolCapabilityRealizationOutcome.REQUIRES_HITL:
        return CapabilityRealizationReasonCode.HUMAN_APPROVAL_REQUIRED
    return CapabilityRealizationReasonCode.DOMAIN_REALIZATION_FAILED


def _failure(
    *,
    request: CapabilityRealizationRequest,
    outcome: CapabilityRealizationOutcome,
    reason_code: CapabilityRealizationReasonCode,
    started_at: datetime,
    reason_detail: str,
) -> CapabilityRealizationResult:
    return CapabilityRealizationResult(
        request_id=request.request_id,
        realization_need_id=request.realization_need.realization_need_id,
        provider_id=TOOL_CAPABILITY_REALIZATION_PROVIDER_ID,
        outcome=outcome,
        reason_code=reason_code,
        capability_identity=request.realization_need.capability_identity,
        started_at=started_at,
        completed_at=datetime.now(tz=UTC),
        reason_detail=reason_detail,
    )


__all__ = [
    "TOOL_CAPABILITY_REALIZATION_PROVIDER_ID",
    "ToolCapabilityRealizationProvider",
]
