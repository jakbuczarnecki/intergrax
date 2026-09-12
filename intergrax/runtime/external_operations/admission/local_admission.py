# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default policy-based external operation admission (R1)."""

from __future__ import annotations

from intergrax.contracts.external_operations.admission import (
    ExternalOperationAdmissionContext,
    OperationAdmissionDecision,
    OperationAdmissionVerdict,
)
from intergrax.contracts.external_operations.intent import (
    ExternalOperationIntent,
    ExternalOperationType,
)
from intergrax.contracts.external_operations.provider import ProviderRiskProfile


class PolicyExternalOperationAdmission:
    """Fail-closed admission — production targets require approval."""

    def evaluate(
        self,
        intent: ExternalOperationIntent,
        context: ExternalOperationAdmissionContext,
    ) -> OperationAdmissionDecision:
        if intent.tenant_id != context.tenant_id:
            return OperationAdmissionDecision(
                verdict=OperationAdmissionVerdict.DENY,
                reason="tenant scope mismatch",
            )
        if context.production_target or _is_production_resource(intent.target_resource):
            if context.human_approval_granted and context.approval_id:
                return OperationAdmissionDecision(
                    verdict=OperationAdmissionVerdict.ALLOW,
                    reason="production resource with recorded human approval",
                    approval_id=context.approval_id,
                    decision_id=context.decision_id,
                )
            return OperationAdmissionDecision(
                verdict=OperationAdmissionVerdict.REQUIRES_APPROVAL,
                reason="production resource",
                decision_id=context.decision_id,
                risk_score=0.85,
            )
        if intent.operation_type in {
            ExternalOperationType.CLOUD_MUTATION,
            ExternalOperationType.CONNECTOR_RESTART,
        }:
            return OperationAdmissionDecision(
                verdict=OperationAdmissionVerdict.REQUIRES_APPROVAL,
                reason=f"high-impact operation type {intent.operation_type}",
                decision_id=context.decision_id,
                risk_score=0.7,
            )
        return OperationAdmissionDecision(
            verdict=OperationAdmissionVerdict.ALLOW,
            reason="policy allow",
            decision_id=context.decision_id,
        )


def risk_requires_approval(profile: ProviderRiskProfile) -> bool:
    return profile in {
        ProviderRiskProfile.HIGH,
        ProviderRiskProfile.PRODUCTION,
    }


def _is_production_resource(target: str) -> bool:
    lowered = target.lower()
    return "production" in lowered or lowered.startswith("prod:")
