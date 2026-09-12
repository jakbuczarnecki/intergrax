# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing workflow orchestrator — lifecycle only, spine executes (SELF-HEALING R2)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from intergrax.contracts.execution_identity import TaskId, mint_task_id, validate_task_id
from intergrax.contracts.external_operations.safety import (
    ExternalOperationAdmissionDeniedError,
    ExternalOperationApprovalRequiredError,
)
from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.decision import SelfHealingDecision, SelfHealingProposedAction
from intergrax.contracts.self_healing.governance import SelfHealingAdmissionContext
from intergrax.contracts.self_healing.workflow.context import SelfHealingWorkflowContext
from intergrax.contracts.self_healing.workflow.errors import (
    PLUGIN_FAILED,
    SelfHealingWorkflowGovernanceError,
    SelfHealingWorkflowPluginFailedError,
    SelfHealingWorkflowStateError,
    SelfHealingWorkflowValidationError,
)
from intergrax.contracts.self_healing.workflow.lifecycle import (
    SelfHealingWorkflowAuditEntry,
    SelfHealingWorkflowState,
    assert_workflow_transition,
    mint_self_healing_workflow_id,
)
from intergrax.contracts.self_healing.workflow.outcome import SelfHealingWorkflowOutcome
from intergrax.contracts.self_healing.workflow.plan import SelfHealingPlan
from intergrax.contracts.self_healing.workflow.plan_builder import SelfHealingPlanBuilder
from intergrax.contracts.self_healing.workflow.step import SelfHealingStep
from intergrax.contracts.self_healing.workflow.validation import ValidationStatus
from intergrax.contracts.self_healing.workflow.registry import (
    SelfHealingPlanBuilderRegistry,
    SelfHealingRollbackRegistry,
    SelfHealingValidationRegistry,
)
from intergrax.runtime.self_healing.orchestrator import GovernedSelfHealingOrchestrator

_WAIT_INTENT = "self_healing.workflow.wait_stabilization"
_VALIDATE_INTENT = "self_healing.workflow.validate"
_ROLLBACK_INTENT = "self_healing.rollback.restore"


@dataclass
class SelfHealingWorkflowOrchestrator:
    """
    Coordinates plan → governance → step delegation → validation → rollback.

    Never calls provider execute APIs directly — reuses ``GovernedSelfHealingOrchestrator``.
    """

    healing_orchestrator: GovernedSelfHealingOrchestrator
    plan_builders: SelfHealingPlanBuilderRegistry
    validation_registry: SelfHealingValidationRegistry
    rollback_registry: SelfHealingRollbackRegistry
    audit_trail: list[SelfHealingWorkflowAuditEntry] = field(default_factory=list)
    spine_attempt_ids: list[str] = field(default_factory=list)
    _workflows: dict[str, SelfHealingWorkflowContext] = field(default_factory=dict)

    def create_from_decision(
        self,
        decision: SelfHealingDecision,
        context: SelfHealingContext,
        *,
        plan_builder: SelfHealingPlanBuilder | None = None,
    ) -> SelfHealingWorkflowContext:
        builder = plan_builder or self.plan_builders.resolve_for_strategy(
            decision.strategy_id,
            tenant_id=context.tenant_id,
        )
        if builder is None:
            raise SelfHealingWorkflowStateError("no plan builder for strategy")
        try:
            plan = builder.build_plan(decision, context)
        except Exception as exc:  # noqa: BLE001 — plugin containment
            raise SelfHealingWorkflowPluginFailedError(
                f"{PLUGIN_FAILED}: plan builder failed: {exc}",
            ) from exc
        return self._start_workflow(decision, context, plan)

    def _start_workflow(
        self,
        decision: SelfHealingDecision,
        context: SelfHealingContext,
        plan: SelfHealingPlan,
    ) -> SelfHealingWorkflowContext:
        now = datetime.now(timezone.utc)
        workflow_id = mint_self_healing_workflow_id()
        wf = SelfHealingWorkflowContext(
            workflow_id=workflow_id,
            tenant_id=context.tenant_id,
            plan=plan,
            decision=decision,
            healing_context=context,
            state=SelfHealingWorkflowState.CREATED,
            successful_step_ids=(),
            failed_step_ids=(),
            evidence_refs=plan.evidence_refs,
            validation_result=None,
            started_at=now,
            updated_at=now,
        )
        self._workflows[workflow_id] = wf
        wf = self._transition(wf, SelfHealingWorkflowState.PLANNED, actor="workflow", reason="plan built")
        return wf

    def run_to_completion(
        self,
        workflow_id: str,
        *,
        admission_context: SelfHealingAdmissionContext,
        task_id: TaskId | None = None,
    ) -> tuple[SelfHealingWorkflowContext, SelfHealingWorkflowOutcome]:
        wf = self._require_workflow(workflow_id)
        if wf.tenant_id != admission_context.tenant_id:
            raise SelfHealingWorkflowStateError("tenant scope mismatch")
        task = task_id or mint_task_id()
        validate_task_id(task)
        wf = self._maybe_wait_approval(wf, admission_context)
        wf = self._transition(wf, SelfHealingWorkflowState.EXECUTING, actor="workflow", reason="execute steps")
        wf = self._execute_steps(wf, admission_context=admission_context, task_id=task)
        wf = self._transition(wf, SelfHealingWorkflowState.VALIDATING, actor="workflow", reason="validate")
        wf = self._run_validation(wf)
        if wf.validation_result and wf.validation_result.status is not ValidationStatus.PASSED:
            wf = self._transition(
                wf,
                SelfHealingWorkflowState.ROLLBACK_REQUIRED,
                actor="workflow",
                reason="validation failed",
            )
            wf = self._transition(wf, SelfHealingWorkflowState.ROLLING_BACK, actor="workflow", reason="rollback")
            wf = self._run_rollback(wf, admission_context=admission_context, task_id=task)
            wf = self._transition(wf, SelfHealingWorkflowState.FAILED, actor="workflow", reason="rollback complete")
        else:
            wf = self._transition(wf, SelfHealingWorkflowState.SUCCEEDED, actor="workflow", reason="validated")
        outcome = self._build_outcome(wf)
        return wf, outcome

    def _maybe_wait_approval(
        self,
        wf: SelfHealingWorkflowContext,
        admission_context: SelfHealingAdmissionContext,
    ) -> SelfHealingWorkflowContext:
        needs = wf.decision.required_approval or wf.healing_context.constraints.production_target
        if needs and not (admission_context.human_approval_granted and admission_context.approval_id):
            wf = self._transition(
                wf,
                SelfHealingWorkflowState.WAITING_APPROVAL,
                actor="governance",
                reason="approval required",
            )
            raise SelfHealingWorkflowGovernanceError("workflow waiting for approval")
        wf = self._transition(wf, SelfHealingWorkflowState.APPROVED, actor="governance", reason="approved")
        return wf

    def _execute_steps(
        self,
        wf: SelfHealingWorkflowContext,
        *,
        admission_context: SelfHealingAdmissionContext,
        task_id: TaskId,
    ) -> SelfHealingWorkflowContext:
        success: list[str] = list(wf.successful_step_ids)
        failed: list[str] = list(wf.failed_step_ids)
        for step in wf.plan.steps:
            if step.operation_intent in {_WAIT_INTENT, _VALIDATE_INTENT}:
                success.append(step.step_id)
                continue
            try:
                self._delegate_step_to_spine(
                    wf,
                    step,
                    admission_context=admission_context,
                    task_id=task_id,
                )
                success.append(step.step_id)
            except (ExternalOperationAdmissionDeniedError, ExternalOperationApprovalRequiredError) as exc:
                failed.append(step.step_id)
                raise SelfHealingWorkflowGovernanceError(str(exc)) from exc
            except Exception:
                failed.append(step.step_id)
                raise
        return self._update_steps(wf, success=tuple(success), failed=tuple(failed))

    def _delegate_step_to_spine(
        self,
        wf: SelfHealingWorkflowContext,
        step: SelfHealingStep,
        *,
        admission_context: SelfHealingAdmissionContext,
        task_id: TaskId,
    ) -> None:
        action = self._action_for_step(wf.decision, step)
        step_decision = replace(
            wf.decision,
            proposed_actions=(action,),
        )
        attempt = self.healing_orchestrator.attempt_execution(
            step_decision,
            tenant_id=wf.tenant_id,
            context=wf.healing_context,
            admission_context=admission_context,
            task_id=task_id,
            production_target=wf.healing_context.constraints.production_target,
        )
        self.spine_attempt_ids.append(attempt.operation_attempt_id)

    def _run_validation(self, wf: SelfHealingWorkflowContext) -> SelfHealingWorkflowContext:
        provider = self.validation_registry.resolve(
            wf.plan.validation_policy_id,
            tenant_id=wf.tenant_id,
        )
        if provider is None:
            raise SelfHealingWorkflowValidationError("validation provider not registered")
        try:
            result = provider.validate(wf)
        except Exception as exc:  # noqa: BLE001
            raise SelfHealingWorkflowPluginFailedError(
                f"{PLUGIN_FAILED}: validation provider failed: {exc}",
            ) from exc
        updated = replace(wf, validation_result=result, updated_at=datetime.now(timezone.utc))
        self._workflows[wf.workflow_id] = updated
        return updated

    def _run_rollback(
        self,
        wf: SelfHealingWorkflowContext,
        *,
        admission_context: SelfHealingAdmissionContext,
        task_id: TaskId,
    ) -> SelfHealingWorkflowContext:
        provider = self.rollback_registry.resolve(
            wf.plan.rollback_policy_id,
            tenant_id=wf.tenant_id,
        )
        if provider is None:
            raise SelfHealingWorkflowValidationError("rollback provider not registered")
        try:
            rollback_plan = provider.plan_rollback(wf)
        except Exception as exc:  # noqa: BLE001
            raise SelfHealingWorkflowPluginFailedError(
                f"{PLUGIN_FAILED}: rollback provider failed: {exc}",
            ) from exc
        for directive in rollback_plan.directives:
            action = SelfHealingProposedAction(
                action_type=_ROLLBACK_INTENT,
                target_resource=directive.target_resource,
                operation_kind=directive.operation_intent,
                provider_id=wf.decision.proposed_actions[0].provider_id,
                rationale=directive.rationale,
            )
            step = SelfHealingStep(
                step_id=directive.directive_id,
                operation_intent=directive.operation_intent,
                required_capability="workflow.rollback",
                sequence_number=0,
                validation_requirements=(),
                rollback_reference=None,
            )
            self._delegate_step_to_spine(
                wf,
                step,
                admission_context=admission_context,
                task_id=task_id,
            )
        return wf

    def _action_for_step(
        self,
        decision: SelfHealingDecision,
        step: SelfHealingStep,
    ) -> SelfHealingProposedAction:
        for action in decision.proposed_actions:
            if action.action_type == step.operation_intent:
                return action
        return decision.proposed_actions[0]

    def _build_outcome(self, wf: SelfHealingWorkflowContext) -> SelfHealingWorkflowOutcome:
        recovery = wf.updated_at - wf.started_at
        rollback = wf.state in {
            SelfHealingWorkflowState.FAILED,
            SelfHealingWorkflowState.ROLLING_BACK,
        } and bool(wf.successful_step_ids)
        return SelfHealingWorkflowOutcome(
            workflow_id=wf.workflow_id,
            strategy_id=wf.decision.strategy_id,
            tenant_id=wf.tenant_id,
            successful_steps=wf.successful_step_ids,
            failed_steps=wf.failed_step_ids,
            rollback_executed=rollback
            or (
                wf.validation_result is not None
                and wf.validation_result.status is not ValidationStatus.PASSED
            ),
            validation_result=wf.validation_result,
            recovery_time=recovery,
            evidence_refs=wf.evidence_refs,
        )

    def _require_workflow(self, workflow_id: str) -> SelfHealingWorkflowContext:
        wf = self._workflows.get(workflow_id)
        if wf is None:
            raise SelfHealingWorkflowStateError("unknown workflow")
        return wf

    def _transition(
        self,
        wf: SelfHealingWorkflowContext,
        target: SelfHealingWorkflowState,
        *,
        actor: str,
        reason: str,
    ) -> SelfHealingWorkflowContext:
        try:
            assert_workflow_transition(wf.state, target)
        except ValueError as exc:
            raise SelfHealingWorkflowStateError(str(exc)) from exc
        now = datetime.now(timezone.utc)
        self.audit_trail.append(
            SelfHealingWorkflowAuditEntry(
                workflow_id=wf.workflow_id,
                tenant_id=wf.tenant_id,
                from_state=wf.state,
                to_state=target,
                actor=actor,
                reason=reason,
                recorded_at=now,
            ),
        )
        updated = replace(wf, state=target, updated_at=now)
        self._workflows[wf.workflow_id] = updated
        return updated

    def _update_steps(
        self,
        wf: SelfHealingWorkflowContext,
        *,
        success: tuple[str, ...],
        failed: tuple[str, ...],
    ) -> SelfHealingWorkflowContext:
        updated = replace(
            wf,
            successful_step_ids=success,
            failed_step_ids=failed,
            updated_at=datetime.now(timezone.utc),
        )
        self._workflows[wf.workflow_id] = updated
        return updated


__all__ = ["SelfHealingWorkflowOrchestrator"]
