# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Healing execution lifecycle — orchestrates, never executes (SELF-HEALING R3)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from intergrax.contracts.execution_identity import TaskId, mint_task_id, validate_task_id
from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.decision import SelfHealingDecision
from intergrax.contracts.self_healing.execution.context import SelfHealingExecutionContext
from intergrax.contracts.self_healing.execution.lifecycle import (
    SelfHealingExecutionLifecycleAuditEntry,
    SelfHealingExecutionLifecycleState,
    assert_execution_lifecycle_transition,
)
from intergrax.contracts.self_healing.governance import SelfHealingAdmissionContext
from intergrax.contracts.self_healing.observation.provider import (
    ObservationResult,
    SelfHealingObservationProvider,
)
from intergrax.contracts.self_healing.validation.decision import ValidationDecisionStatus
from intergrax.contracts.self_healing.workflow.context import SelfHealingWorkflowContext
from intergrax.contracts.self_healing.workflow.errors import SelfHealingWorkflowGovernanceError
from intergrax.contracts.self_healing.workflow.lifecycle import SelfHealingWorkflowState
from intergrax.contracts.self_healing.workflow.outcome import SelfHealingWorkflowOutcome
from intergrax.runtime.self_healing.lifecycle.default_observation import PlatformExecutionObservationProvider
from intergrax.runtime.self_healing.lifecycle.rollback_coordinator import SelfHealingRollbackCoordinator
from intergrax.runtime.self_healing.lifecycle.validation_pipeline import SelfHealingValidationPipeline
from intergrax.runtime.self_healing.workflow.orchestrator import SelfHealingWorkflowOrchestrator


@dataclass
class _LifecycleSession:
    workflow: SelfHealingWorkflowContext
    state: SelfHealingExecutionLifecycleState
    execution_context: SelfHealingExecutionContext | None = None
    observation: ObservationResult | None = None


@dataclass
class SelfHealingLifecycleEngine:
    workflow_orchestrator: SelfHealingWorkflowOrchestrator
    validation_pipeline: SelfHealingValidationPipeline
    rollback_coordinator: SelfHealingRollbackCoordinator
    observation_provider: SelfHealingObservationProvider = field(
        default_factory=PlatformExecutionObservationProvider,
    )
    audit_trail: list[SelfHealingExecutionLifecycleAuditEntry] = field(default_factory=list)
    _sessions: dict[str, _LifecycleSession] = field(default_factory=dict)

    def start_from_decision(
        self,
        decision: SelfHealingDecision,
        context: SelfHealingContext,
    ) -> SelfHealingExecutionContext:
        wf = self.workflow_orchestrator.create_from_decision(decision, context)
        session = _LifecycleSession(
            workflow=wf,
            state=SelfHealingExecutionLifecycleState.CREATED,
        )
        self._sessions[wf.workflow_id] = session
        self._transition(session, SelfHealingExecutionLifecycleState.APPROVAL_PENDING, actor="lifecycle")
        exec_ctx = SelfHealingExecutionContext(
            workflow_id=wf.workflow_id,
            plan_id=wf.plan.plan_id,
            strategy_id=decision.strategy_id,
            tenant_id=wf.tenant_id,
            execution_ids=(),
            operation_attempt_ids=(),
            evidence_refs=wf.evidence_refs,
            created_at=datetime.now(timezone.utc),
        )
        session.execution_context = exec_ctx
        return exec_ctx

    def run_to_completion(
        self,
        workflow_id: str,
        *,
        admission_context: SelfHealingAdmissionContext,
        task_id: TaskId | None = None,
    ) -> tuple[SelfHealingExecutionContext, SelfHealingWorkflowOutcome]:
        session = self._require_session(workflow_id)
        if session.workflow.tenant_id != admission_context.tenant_id:
            raise SelfHealingWorkflowGovernanceError("tenant scope mismatch")
        task = task_id or mint_task_id()
        validate_task_id(task)

        wf = session.workflow
        try:
            wf = self.workflow_orchestrator._maybe_wait_approval(  # noqa: SLF001
                wf,
                admission_context,
            )
        except SelfHealingWorkflowGovernanceError:
            self._transition(session, SelfHealingExecutionLifecycleState.APPROVAL_PENDING, actor="governance")
            raise
        session.workflow = wf
        self._transition(session, SelfHealingExecutionLifecycleState.APPROVED, actor="governance")
        self._transition(session, SelfHealingExecutionLifecycleState.EXECUTION_REQUESTED, actor="lifecycle")
        self._transition(session, SelfHealingExecutionLifecycleState.EXECUTING, actor="lifecycle")

        wf = self.workflow_orchestrator._transition(  # noqa: SLF001
            wf,
            SelfHealingWorkflowState.EXECUTING,
            actor="lifecycle",
            reason="execute steps",
        )
        session.workflow = wf
        self.workflow_orchestrator.spine_attempt_ids.clear()
        audit_before = len(self.workflow_orchestrator.healing_orchestrator.audit_records)
        wf = self.workflow_orchestrator._execute_steps(  # noqa: SLF001
            wf,
            admission_context=admission_context,
            task_id=task,
        )
        session.workflow = wf
        exec_ctx = self._correlate_execution_context(session, audit_before)
        session.execution_context = exec_ctx

        self._transition(session, SelfHealingExecutionLifecycleState.OBSERVING, actor="lifecycle")
        observation = self.observation_provider.observe(exec_ctx)
        session.observation = observation

        wf = self.workflow_orchestrator._transition(  # noqa: SLF001
            wf,
            SelfHealingWorkflowState.VALIDATING,
            actor="lifecycle",
            reason="validate",
        )
        session.workflow = wf
        self._transition(session, SelfHealingExecutionLifecycleState.VALIDATING, actor="lifecycle")
        decision = self.validation_pipeline.evaluate(
            wf,
            execution_context=exec_ctx,
            observation=observation,
        )
        legacy = SelfHealingRollbackCoordinator.to_legacy_validation_result(
            decision.status,
            evidence_refs=decision.evidence_refs,
            confidence=decision.confidence,
            explanation=decision.explanation,
        )
        wf = replace(wf, validation_result=legacy, updated_at=datetime.now(timezone.utc))
        session.workflow = wf
        self.workflow_orchestrator._workflows[wf.workflow_id] = wf  # noqa: SLF001

        if decision.status is ValidationDecisionStatus.FAILED:
            wf = self.workflow_orchestrator._transition(  # noqa: SLF001
                wf,
                SelfHealingWorkflowState.ROLLBACK_REQUIRED,
                actor="lifecycle",
                reason="validation failed",
            )
            session.workflow = wf
            self._transition(session, SelfHealingExecutionLifecycleState.ROLLBACK_PENDING, actor="lifecycle")
            wf = self.workflow_orchestrator._transition(  # noqa: SLF001
                wf,
                SelfHealingWorkflowState.ROLLING_BACK,
                actor="lifecycle",
                reason="rollback",
            )
            session.workflow = wf
            wf = self.rollback_coordinator.coordinate_rollback(
                wf,
                admission_context=admission_context,
                task_id=task,
            )
            session.workflow = wf
            wf = self.workflow_orchestrator._transition(  # noqa: SLF001
                wf,
                SelfHealingWorkflowState.FAILED,
                actor="lifecycle",
                reason="rollback complete",
            )
            session.workflow = wf
            self._transition(session, SelfHealingExecutionLifecycleState.ROLLED_BACK, actor="lifecycle")
            outcome = self.workflow_orchestrator._build_outcome(wf)  # noqa: SLF001
            return exec_ctx, outcome

        if decision.status is ValidationDecisionStatus.PASSED:
            wf = self.workflow_orchestrator._transition(  # noqa: SLF001
                wf,
                SelfHealingWorkflowState.SUCCEEDED,
                actor="lifecycle",
                reason="validated",
            )
            session.workflow = wf
            self._transition(session, SelfHealingExecutionLifecycleState.COMPLETED, actor="lifecycle")
        else:
            self._transition(session, SelfHealingExecutionLifecycleState.FAILED, actor="lifecycle")

        outcome = self.workflow_orchestrator._build_outcome(wf)  # noqa: SLF001
        return exec_ctx, outcome

    def _correlate_execution_context(
        self,
        session: _LifecycleSession,
        audit_before: int,
    ) -> SelfHealingExecutionContext:
        base = session.execution_context
        if base is None:
            raise ValueError("execution context missing")
        records = self.workflow_orchestrator.healing_orchestrator.audit_records[audit_before:]
        execution_ids: list[str] = []
        for record in records:
            if record.execution_id:
                execution_ids.append(record.execution_id)
        attempt_ids = list(self.workflow_orchestrator.spine_attempt_ids)
        return SelfHealingExecutionContext(
            workflow_id=base.workflow_id,
            plan_id=base.plan_id,
            strategy_id=base.strategy_id,
            tenant_id=base.tenant_id,
            execution_ids=tuple(execution_ids),
            operation_attempt_ids=tuple(attempt_ids),
            evidence_refs=base.evidence_refs,
            created_at=base.created_at,
            metadata=base.metadata,
        )

    def _require_session(self, workflow_id: str) -> _LifecycleSession:
        session = self._sessions.get(workflow_id)
        if session is None:
            raise ValueError("unknown lifecycle session")
        return session

    def _transition(
        self,
        session: _LifecycleSession,
        target: SelfHealingExecutionLifecycleState,
        *,
        actor: str,
        reason: str = "",
    ) -> None:
        try:
            assert_execution_lifecycle_transition(session.state, target)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        now = datetime.now(timezone.utc)
        refs = session.execution_context.evidence_refs if session.execution_context else ()
        self.audit_trail.append(
            SelfHealingExecutionLifecycleAuditEntry(
                workflow_id=session.workflow.workflow_id,
                tenant_id=session.workflow.tenant_id,
                from_state=session.state,
                to_state=target,
                actor=actor,
                reason=reason or target.value,
                recorded_at=now,
                evidence_refs=refs,
            ),
        )
        session.state = target


__all__ = ["SelfHealingLifecycleEngine"]
