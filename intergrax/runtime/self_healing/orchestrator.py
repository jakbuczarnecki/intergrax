# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Governed self-healing orchestration — reuses external operation execution (SELF-HEALING R1)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from intergrax.contracts.self_healing.action_provider import SelfHealingActionProvider
from intergrax.contracts.self_healing.audit import SelfHealingAuditRecord
from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.decision import SelfHealingDecision, SelfHealingProposedAction
from intergrax.contracts.self_healing.governance import (
    SelfHealingAdmissionContext,
    SelfHealingAdmissionGate,
    SelfHealingAdmissionVerdict,
)
from intergrax.contracts.self_healing.result import SelfHealingExecutionOutcome
from intergrax.contracts.self_healing.safety import assert_decision_has_no_execution_surface
from intergrax.runtime.external_operations.admission.execution_gate import (
    ExternalOperationExecutionGate,
)
from intergrax.runtime.self_healing.governance_bridge import resolve_self_healing_governance_chain
from intergrax.runtime.self_healing.safety_evaluator import SelfHealingSafetyEvaluator


@dataclass
class GovernedSelfHealingOrchestrator:
    admission_gate: SelfHealingAdmissionGate
    external_gate: ExternalOperationExecutionGate
    external_admission: ExternalOperationAdmission
    providers: dict[str, SelfHealingActionProvider]
    safety_evaluator: SelfHealingSafetyEvaluator = field(default_factory=SelfHealingSafetyEvaluator)
    audit_records: list[SelfHealingAuditRecord] = field(default_factory=list)

    def translate_action(
        self,
        decision: SelfHealingDecision,
        action: SelfHealingProposedAction,
        *,
        tenant_id: str,
        task_id: TaskId,
        requested_by: str = "self_healing.governance",
    ) -> ExternalOperationIntent:
        validate_task_id(task_id)
        provider = self._resolve_provider(action)
        return provider.translate(
            decision,
            action,
            tenant_id=tenant_id,
            task_id=task_id,
            requested_by=requested_by,
        )

    def attempt_execution(
        self,
        decision: SelfHealingDecision,
        *,
        tenant_id: str,
        context: SelfHealingContext,
        admission_context: SelfHealingAdmissionContext,
        task_id: TaskId,
        requested_by: str = "self_healing.governance",
        provider_id: str | None = None,
        production_target: bool = False,
    ) -> ExternalOperationAttempt:
        assert_decision_has_no_execution_surface(decision)
        if tenant_id != context.tenant_id or tenant_id != admission_context.tenant_id:
            raise ExternalOperationAdmissionDeniedError("tenant scope mismatch")
        if not decision.proposed_actions:
            raise ExternalOperationAdmissionDeniedError("decision has no proposed actions")
        if production_target and not admission_context.production_target:
            admission_context = replace(admission_context, production_target=True)
        action = decision.proposed_actions[0]
        safety = self.safety_evaluator.evaluate(decision, context)
        if not safety.may_proceed_to_governance:
            self._record_audit(
                decision,
                tenant_id=tenant_id,
                outcome=SelfHealingExecutionOutcome.DENIED,
                admission_reason=safety.reason,
            )
            raise ExternalOperationAdmissionDeniedError(safety.reason)
        intent = self.translate_action(
            decision,
            action,
            tenant_id=tenant_id,
            task_id=task_id,
            requested_by=requested_by,
        )
        provider = self._resolve_provider(action)
        chain = resolve_self_healing_governance_chain(
            decision=decision,
            admission_gate=self.admission_gate,
            admission_context=admission_context,
            external_intent=intent,
            external_admission=self.external_admission,
            external_context=ExternalOperationAdmissionContext(
                tenant_id=tenant_id,
                provider_id=provider_id or provider.provider_id,
                production_target=production_target,
                decision_id=admission_context.decision_id,
                human_approval_granted=admission_context.human_approval_granted,
                approval_id=admission_context.approval_id,
            ),
        )
        admission = chain.self_healing_admission
        if admission.verdict is SelfHealingAdmissionVerdict.DENY:
            self._record_audit(
                decision,
                tenant_id=tenant_id,
                outcome=SelfHealingExecutionOutcome.DENIED,
                admission_reason=admission.reason,
            )
            raise ExternalOperationAdmissionDeniedError(admission.reason)
        if admission.verdict is SelfHealingAdmissionVerdict.REQUIRES_APPROVAL:
            if not (admission_context.human_approval_granted and admission_context.approval_id):
                self._record_audit(
                    decision,
                    tenant_id=tenant_id,
                    outcome=SelfHealingExecutionOutcome.PENDING,
                    admission_reason=admission.reason,
                )
                raise ExternalOperationApprovalRequiredError(admission.reason)
        if not chain.may_execute:
            self._record_audit(
                decision,
                tenant_id=tenant_id,
                outcome=SelfHealingExecutionOutcome.DENIED,
                admission_reason="external admission blocked execution",
                external_operation_id=intent.intent_id,
            )
            raise ExternalOperationAdmissionDeniedError("external admission blocked execution")
        attempt = self.external_gate.admit_intent(
            intent,
            context=ExternalOperationAdmissionContext(
                tenant_id=tenant_id,
                provider_id=provider_id or provider.provider_id,
                production_target=production_target,
                decision_id=admission_context.decision_id,
                human_approval_granted=admission_context.human_approval_granted,
                approval_id=admission_context.approval_id,
            ),
            provider_id=provider_id or provider.provider_id,
        )
        self._record_audit(
            decision,
            tenant_id=tenant_id,
            outcome=SelfHealingExecutionOutcome.SUCCEEDED,
            admission_reason=admission.reason,
            approval_refs=(admission_context.approval_id,)
            if admission_context.approval_id
            else (),
            external_operation_id=intent.intent_id,
            execution_id=str(attempt.execution_id) if attempt.execution_id else None,
            action_type=action.action_type,
        )
        return attempt

    def _resolve_provider(self, action: SelfHealingProposedAction) -> SelfHealingActionProvider:
        provider = self.providers.get(action.provider_id)
        if provider is not None and action.action_type in provider.supported_action_types:
            return provider
        for candidate in self.providers.values():
            if action.action_type in candidate.supported_action_types:
                return candidate
        raise KeyError(f"no self-healing action provider for {action.action_type}")

    def _record_audit(
        self,
        decision: SelfHealingDecision,
        *,
        tenant_id: str,
        outcome: SelfHealingExecutionOutcome,
        admission_reason: str,
        approval_refs: tuple[str, ...] = (),
        external_operation_id: str | None = None,
        execution_id: str | None = None,
        action_type: str = "",
    ) -> SelfHealingAuditRecord:
        record = SelfHealingAuditRecord(
            decision_id=decision.decision_id,
            strategy_id=decision.strategy_id,
            tenant_id=tenant_id,
            evidence_refs=decision.evidence_refs,
            approval_refs=approval_refs,
            external_operation_id=external_operation_id,
            execution_id=execution_id,
            outcome=outcome,
            recorded_at=datetime.now(timezone.utc),
            action_type=action_type or (
                decision.proposed_actions[0].action_type if decision.proposed_actions else ""
            ),
            admission_reason=admission_reason,
        )
        self.audit_records.append(record)
        return record


__all__ = ["GovernedSelfHealingOrchestrator"]
