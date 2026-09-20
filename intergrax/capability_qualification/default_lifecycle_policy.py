# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default lifecycle disposition from qualification outcomes (UCA-4)."""

from __future__ import annotations

from intergrax.contracts.capability_qualification.lifecycle_decision import (
    CapabilityQualificationLifecycleDecision,
    CapabilityQualificationLifecycleOutcome,
    CapabilityQualificationLifecycleReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)


class DefaultCapabilityQualificationLifecyclePolicy:
    """Maps qualification facts to lifecycle disposition — no domain mutation."""

    def decide(
        self,
        *,
        request: CapabilityQualificationRequest,
        qualification_result: CapabilityQualificationResult,
    ) -> CapabilityQualificationLifecycleDecision:
        del request
        outcome = qualification_result.outcome
        if outcome is CapabilityQualificationOutcome.QUALIFIED:
            return CapabilityQualificationLifecycleDecision(
                outcome=CapabilityQualificationLifecycleOutcome.ACCEPT,
                reason_code=CapabilityQualificationLifecycleReasonCode.QUALIFICATION_ACCEPTED,
            )
        if outcome is CapabilityQualificationOutcome.REJECTED:
            return CapabilityQualificationLifecycleDecision(
                outcome=CapabilityQualificationLifecycleOutcome.REJECT,
                reason_code=CapabilityQualificationLifecycleReasonCode.QUALIFICATION_REJECTED,
                reason_detail=qualification_result.reason_detail,
            )
        if outcome is CapabilityQualificationOutcome.BLOCKED:
            return CapabilityQualificationLifecycleDecision(
                outcome=CapabilityQualificationLifecycleOutcome.QUARANTINE,
                reason_code=CapabilityQualificationLifecycleReasonCode.QUALIFICATION_BLOCKED,
                reason_detail=qualification_result.reason_detail,
            )
        if outcome is CapabilityQualificationOutcome.REQUIRES_HITL:
            return CapabilityQualificationLifecycleDecision(
                outcome=CapabilityQualificationLifecycleOutcome.REQUIRES_HITL,
                reason_code=CapabilityQualificationLifecycleReasonCode.HUMAN_REVIEW_REQUIRED,
                reason_detail=qualification_result.reason_detail,
            )
        if outcome is CapabilityQualificationOutcome.NO_PROVIDER:
            return CapabilityQualificationLifecycleDecision(
                outcome=CapabilityQualificationLifecycleOutcome.RETAIN_FOR_REVIEW,
                reason_code=CapabilityQualificationLifecycleReasonCode.NO_QUALIFICATION_PROVIDER,
                reason_detail=qualification_result.reason_detail,
            )
        if outcome is CapabilityQualificationOutcome.CONFLICT:
            return CapabilityQualificationLifecycleDecision(
                outcome=CapabilityQualificationLifecycleOutcome.REQUIRES_HITL,
                reason_code=CapabilityQualificationLifecycleReasonCode.SELECTION_CONFLICT,
                reason_detail=qualification_result.reason_detail,
            )
        if outcome in {
            CapabilityQualificationOutcome.FAILED,
            CapabilityQualificationOutcome.NOT_SUPPORTED,
        }:
            return CapabilityQualificationLifecycleDecision(
                outcome=CapabilityQualificationLifecycleOutcome.DISCARD,
                reason_code=CapabilityQualificationLifecycleReasonCode.PROVIDER_FAILED,
                reason_detail=qualification_result.reason_detail,
            )
        return CapabilityQualificationLifecycleDecision(
            outcome=CapabilityQualificationLifecycleOutcome.RETAIN_FOR_REVIEW,
            reason_code=CapabilityQualificationLifecycleReasonCode.PROVIDER_UNAVAILABLE,
            reason_detail=qualification_result.reason_detail,
        )


__all__ = ["DefaultCapabilityQualificationLifecyclePolicy"]
