# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Governed preventive action orchestration — reuses external operation execution (PREVENTIVE R7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from intergrax.contracts.execution_identity import TaskId, validate_task_id
from intergrax.contracts.external_operations.admission import (
    ExternalOperationAdmission,
    ExternalOperationAdmissionContext,
)
from intergrax.contracts.external_operations.attempt import ExternalOperationAttempt
from intergrax.contracts.external_operations.intent import ExternalOperationIntent
from intergrax.contracts.external_operations.safety import (
    ExternalOperationAdmissionDeniedError,
    ExternalOperationApprovalRequiredError,
)
from intergrax.contracts.preventive.actions.admission import (
    PreventiveActionAdmissionContext,
    PreventiveActionAdmissionGate,
    PreventiveActionAdmissionVerdict,
)
from intergrax.contracts.preventive.actions.audit import (
    PreventiveActionAuditRecord,
    PreventiveActionOutcome,
)
from intergrax.contracts.preventive.actions.lifecycle import (
    PreventiveActionLifecycleState,
    assert_preventive_action_lifecycle_transition,
)
from intergrax.contracts.preventive.actions.proposal import (
    PreventiveActionProposal,
    mint_preventive_action_proposal_id,
)
from intergrax.contracts.preventive.actions.provider import PreventiveActionProvider
from intergrax.contracts.preventive.actions.safety import assert_proposal_has_no_execution_surface
from intergrax.runtime.external_operations.admission.execution_gate import (
    ExternalOperationExecutionGate,
)
from intergrax.runtime.prevention.actions.governance_bridge import (
    resolve_preventive_governance_chain,
)


@dataclass
class GovernedPreventiveActionOrchestrator:
    """
    Coordinates proposal, admission, translation, and external execution gate.

    Never executes without passing preventive + external admission.
    """

    preventive_gate: PreventiveActionAdmissionGate
    external_gate: ExternalOperationExecutionGate
    external_admission: ExternalOperationAdmission
    providers: dict[str, PreventiveActionProvider]
    audit_records: list[PreventiveActionAuditRecord] = field(default_factory=list)
    lifecycle_by_proposal: dict[str, PreventiveActionLifecycleState] = field(
        default_factory=dict,
    )

    def mint_proposal(
        self,
        *,
        tenant_id: str,
        risk_signal_refs: tuple[str, ...],
        recommendation_refs: tuple[str, ...],
        action_type: str,
        target_resource: str,
        justification: str,
        confidence: float,
    ) -> PreventiveActionProposal:
        proposal = PreventiveActionProposal(
            proposal_id=mint_preventive_action_proposal_id(),
            tenant_id=tenant_id,
            risk_signal_refs=risk_signal_refs,
            recommendation_refs=recommendation_refs,
            action_type=action_type,
            target_resource=target_resource,
            justification=justification,
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
        )
        assert_proposal_has_no_execution_surface(proposal)
        self.lifecycle_by_proposal[proposal.proposal_id] = PreventiveActionLifecycleState.PROPOSED
        return proposal

    def translate_proposal(
        self,
        proposal: PreventiveActionProposal,
        *,
        task_id: TaskId,
        requested_by: str = "preventive.governance",
    ) -> ExternalOperationIntent:
        validate_task_id(task_id)
        provider = self._resolve_provider(proposal.action_type)
        return provider.translate(proposal, task_id=task_id, requested_by=requested_by)

    def attempt_execution(
        self,
        proposal: PreventiveActionProposal,
        *,
        preventive_context: PreventiveActionAdmissionContext,
        task_id: TaskId,
        requested_by: str = "preventive.governance",
        provider_id: str | None = None,
        production_target: bool = False,
    ) -> ExternalOperationAttempt:
        provider = self._resolve_provider(proposal.action_type)
        intent = provider.translate(proposal, task_id=task_id, requested_by=requested_by)
        chain = resolve_preventive_governance_chain(
            proposal=proposal,
            preventive_gate=self.preventive_gate,
            preventive_context=preventive_context,
            external_intent=intent,
            external_admission=self.external_admission,
            external_context=ExternalOperationAdmissionContext(
                tenant_id=proposal.tenant_id,
                provider_id=provider_id or provider.provider_id,
                production_target=production_target,
                decision_id=preventive_context.decision_id,
                human_approval_granted=preventive_context.human_approval_granted,
                approval_id=preventive_context.approval_id,
            ),
        )
        admission = chain.preventive_admission
        if admission.verdict is PreventiveActionAdmissionVerdict.DENY:
            self._record_audit(
                proposal,
                outcome=PreventiveActionOutcome.DENIED,
                admission_reason=admission.reason,
            )
            self._transition(proposal.proposal_id, PreventiveActionLifecycleState.REJECTED)
            raise ExternalOperationAdmissionDeniedError(admission.reason)
        if admission.verdict is PreventiveActionAdmissionVerdict.REQUIRES_APPROVAL:
            if not (
                preventive_context.human_approval_granted and preventive_context.approval_id
            ):
                self._transition(
                    proposal.proposal_id,
                    PreventiveActionLifecycleState.WAITING_APPROVAL,
                )
                self._record_audit(
                    proposal,
                    outcome=PreventiveActionOutcome.PENDING,
                    admission_reason=admission.reason,
                    approval_refs=(),
                )
                raise ExternalOperationApprovalRequiredError(admission.reason)
        if not chain.may_execute:
            self._record_audit(
                proposal,
                outcome=PreventiveActionOutcome.DENIED,
                admission_reason="external admission blocked execution",
                external_operation_id=intent.intent_id,
            )
            self._transition(proposal.proposal_id, PreventiveActionLifecycleState.FAILED)
            raise ExternalOperationAdmissionDeniedError("external admission blocked execution")
        self._transition(proposal.proposal_id, PreventiveActionLifecycleState.APPROVED)
        attempt = self.external_gate.admit_intent(
            intent,
            context=ExternalOperationAdmissionContext(
                tenant_id=proposal.tenant_id,
                provider_id=provider_id or provider.provider_id,
                production_target=production_target,
                decision_id=preventive_context.decision_id,
                human_approval_granted=preventive_context.human_approval_granted,
                approval_id=preventive_context.approval_id,
            ),
            provider_id=provider_id or provider.provider_id,
        )
        self._transition(proposal.proposal_id, PreventiveActionLifecycleState.EXECUTED)
        self._record_audit(
            proposal,
            outcome=PreventiveActionOutcome.SUCCEEDED,
            admission_reason=admission.reason,
            approval_refs=(preventive_context.approval_id,)
            if preventive_context.approval_id
            else (),
            external_operation_id=intent.intent_id,
            execution_id=str(attempt.execution_id) if attempt.execution_id else None,
        )
        return attempt

    def _resolve_provider(self, action_type: str) -> PreventiveActionProvider:
        for provider in self.providers.values():
            if action_type in provider.supported_action_types:
                return provider
        raise KeyError(f"no preventive action provider for {action_type}")

    def _transition(self, proposal_id: str, target: PreventiveActionLifecycleState) -> None:
        current = self.lifecycle_by_proposal.get(
            proposal_id,
            PreventiveActionLifecycleState.PROPOSED,
        )
        assert_preventive_action_lifecycle_transition(current, target)
        self.lifecycle_by_proposal[proposal_id] = target

    def _record_audit(
        self,
        proposal: PreventiveActionProposal,
        *,
        outcome: PreventiveActionOutcome,
        admission_reason: str,
        approval_refs: tuple[str, ...] = (),
        external_operation_id: str | None = None,
        execution_id: str | None = None,
    ) -> PreventiveActionAuditRecord:
        record = PreventiveActionAuditRecord(
            proposal_id=proposal.proposal_id,
            tenant_id=proposal.tenant_id,
            risk_signal_refs=proposal.risk_signal_refs,
            recommendation_refs=proposal.recommendation_refs,
            approval_refs=approval_refs,
            external_operation_id=external_operation_id,
            execution_id=execution_id,
            outcome=outcome,
            recorded_at=datetime.now(timezone.utc),
            action_type=proposal.action_type,
            admission_reason=admission_reason,
        )
        self.audit_records.append(record)
        return record


__all__ = ["GovernedPreventiveActionOrchestrator"]
