# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Recovery lifecycle execution handoff — governance gate, port invoke, no direct mutation."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.execution_lifecycle_port import (
    ExecutionLifecyclePort,
    RecoveryLifecycleIntent,
)
from intergrax.contracts.enterprise_reliability.governance_decision import GovernanceDisposition
from intergrax.contracts.enterprise_reliability.recovery_decision import RecoveryLifecycleAction
from intergrax.contracts.enterprise_reliability.recovery_lifecycle_handoff import (
    LifecycleHandoffDisposition,
    RecoveryLifecycleHandoffRequest,
    RecoveryLifecycleHandoffResult,
    governance_result_ref,
    recovery_lifecycle_decision_context_refs,
)
from intergrax.runtime.enterprise_reliability.governance_orchestration import (
    ExternalEffectGovernanceEvaluation,
)


def _handoff_request_from_evaluation(
    *,
    evaluation: ExternalEffectGovernanceEvaluation,
    execution_ref: str,
    tenant_id: str,
    governance_result_ref_value: str | None,
) -> RecoveryLifecycleHandoffRequest:
    evidence = evaluation.evidence
    correlation_id = evaluation.state.correlation_id
    contract_id = evaluation.contract_id
    return RecoveryLifecycleHandoffRequest(
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        contract_id=contract_id,
        execution_ref=execution_ref,
        lifecycle_action=evaluation.recovery_decision.action,
        decision_context=recovery_lifecycle_decision_context_refs(
            evidence_ref=evidence.evidence_ref,
            correlation_id=correlation_id,
            contract_id=contract_id,
        ),
        governance_result_ref=governance_result_ref_value,
    )


def _blocked_result(
    *,
    evaluation: ExternalEffectGovernanceEvaluation,
    execution_ref: str,
    tenant_id: str,
    disposition: LifecycleHandoffDisposition,
    rationale: str,
    governance_token: str,
) -> RecoveryLifecycleHandoffResult:
    request = _handoff_request_from_evaluation(
        evaluation=evaluation,
        execution_ref=execution_ref,
        tenant_id=tenant_id,
        governance_result_ref_value=governance_result_ref(
            correlation_id=evaluation.state.correlation_id,
            contract_id=evaluation.contract_id,
            disposition_token=governance_token,
        ),
    )
    return RecoveryLifecycleHandoffResult(
        disposition=disposition,
        request=request,
        rationale=rationale,
    )


def handoff_recovery_lifecycle_to_execution(
    *,
    governance_evaluation: ExternalEffectGovernanceEvaluation,
    execution_ref: str,
    tenant_id: str,
    lifecycle_port: ExecutionLifecyclePort | None,
) -> RecoveryLifecycleHandoffResult:
    """
    Validate governance and send lifecycle intent to the execution subsystem.

    Does not pause, resume, terminate, or otherwise mutate execution state directly.
    """
    evaluation = governance_evaluation
    governance = evaluation.governance_decision
    correlation_id = evaluation.state.correlation_id
    contract_id = evaluation.contract_id

    if governance.disposition is not GovernanceDisposition.ALLOW:
        if evaluation.recovery_decision.action is RecoveryLifecycleAction.ESCALATE:
            blocked_disposition = LifecycleHandoffDisposition.ESCALATED
            governance_token = "escalated"
            rationale = evaluation.recovery_decision.rationale or "recovery_escalate"
        elif governance.disposition is GovernanceDisposition.APPROVAL_REQUIRED:
            blocked_disposition = LifecycleHandoffDisposition.APPROVAL_REQUIRED
            governance_token = "approval_required"
            rationale = governance.rationale or "governance_approval_required"
        else:
            blocked_disposition = LifecycleHandoffDisposition.BLOCKED
            governance_token = "deny"
            rationale = governance.rationale or "governance_deny"
        return _blocked_result(
            evaluation=evaluation,
            execution_ref=execution_ref,
            tenant_id=tenant_id,
            disposition=blocked_disposition,
            rationale=rationale,
            governance_token=governance_token,
        )

    if lifecycle_port is None:
        request = _handoff_request_from_evaluation(
            evaluation=evaluation,
            execution_ref=execution_ref,
            tenant_id=tenant_id,
            governance_result_ref_value=governance_result_ref(
                correlation_id=correlation_id,
                contract_id=contract_id,
                disposition_token="allow",
            ),
        )
        return RecoveryLifecycleHandoffResult(
            disposition=LifecycleHandoffDisposition.PORT_UNAVAILABLE,
            request=request,
            rationale="execution_lifecycle_port_unavailable",
        )

    request = _handoff_request_from_evaluation(
        evaluation=evaluation,
        execution_ref=execution_ref,
        tenant_id=tenant_id,
        governance_result_ref_value=governance_result_ref(
            correlation_id=correlation_id,
            contract_id=contract_id,
            disposition_token="allow",
        ),
    )
    intent = RecoveryLifecycleIntent(
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        contract_id=contract_id,
        decision=evaluation.recovery_decision,
    )
    lifecycle_port.apply_recovery_lifecycle_intent(intent)
    return RecoveryLifecycleHandoffResult(
        disposition=LifecycleHandoffDisposition.HANDED_OFF,
        request=request,
        rationale="lifecycle_intent_applied",
    )


__all__ = [
    "handoff_recovery_lifecycle_to_execution",
]
