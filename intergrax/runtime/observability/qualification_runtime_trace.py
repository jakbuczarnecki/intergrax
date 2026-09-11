# © Artur Czarnecki. All rights reserved.

"""Emit canonical runtime diagnostics into the active task trace lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    reset_active_execution_identity,
    validate_run_id,
)
from intergrax.runtime.nexus.tracing.execution.evaluator_model_attempt import (
    EvaluatorModelAttemptDiagV1,
)
from intergrax.runtime.diagnostics.completion_alignment_diag import CompletionAlignmentDiagV1
from intergrax.runtime.nexus.tracing.execution.reconciliation_phase import (
    ReconciliationPhaseDiagV1,
    ReconciliationPhaseValue,
)
from intergrax.runtime.nexus.tracing.trace_models import (
    TraceComponent,
    TraceLevel,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_trace import PersistingTaskTraceEmitter, TaskTraceEmitter

EVALUATOR_MODEL_ATTEMPT_STEP = "evaluator_loop.model_attempt"
RECONCILIATION_PHASE_STEP = "completion.reconciliation_phase"
COMPLETION_ALIGNMENT_STEP = "completion.alignment"

O1_SUPPORTED_TRACE_SCHEMA_IDS: tuple[str, ...] = (
    EvaluatorModelAttemptDiagV1.schema_id(),
    ReconciliationPhaseDiagV1.schema_id(),
)

O2_SUPPORTED_TRACE_SCHEMA_IDS: tuple[str, ...] = (
    *O1_SUPPORTED_TRACE_SCHEMA_IDS,
    CompletionAlignmentDiagV1.schema_id(),
)


class RuntimeDiagnosticTracePort(Protocol):
    def emit_evaluator_model_attempt(
        self,
        *,
        run_id: RunId,
        node_id: str,
        attempt_index: int,
        max_iterations: int,
    ) -> None: ...

    def emit_reconciliation_phase(
        self,
        *,
        run_id: RunId,
        validation_invalid: bool,
        entered_reconciliation: bool,
        phase: ReconciliationPhaseValue,
    ) -> None: ...

    def emit_completion_alignment(
        self,
        *,
        payload: CompletionAlignmentDiagV1,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class TaskTraceRuntimeDiagnosticPort:
    """Graph- or scenario-scoped port backed by the active ``TaskTraceEmitter``."""

    trace_emitter: TaskTraceEmitter
    task: Task

    def emit_evaluator_model_attempt(
        self,
        *,
        run_id: RunId,
        node_id: str,
        attempt_index: int,
        max_iterations: int,
    ) -> None:
        payload = EvaluatorModelAttemptDiagV1(
            run_id=str(validate_run_id(run_id)),
            node_id=node_id,
            attempt_index=attempt_index,
            max_iterations=max_iterations,
        )
        self.trace_emitter.emit_trace_step(
            self.task,
            component=TraceComponent.RUNTIME,
            step=EVALUATOR_MODEL_ATTEMPT_STEP,
            message="evaluator loop model attempt",
            level=TraceLevel.INFO,
            payload=payload,
            extra_tags={"node_id": node_id},
        )

    def emit_reconciliation_phase(
        self,
        *,
        run_id: RunId,
        validation_invalid: bool,
        entered_reconciliation: bool,
        phase: ReconciliationPhaseValue,
    ) -> None:
        payload = ReconciliationPhaseDiagV1(
            run_id=str(validate_run_id(run_id)),
            validation_invalid=validation_invalid,
            entered_reconciliation=entered_reconciliation,
            phase=phase,
        )
        self.trace_emitter.emit_trace_step(
            self.task,
            component=TraceComponent.RUNTIME,
            step=RECONCILIATION_PHASE_STEP,
            message="completion reconciliation phase",
            level=TraceLevel.INFO,
            payload=payload,
        )

    def emit_completion_alignment(
        self,
        *,
        payload: CompletionAlignmentDiagV1,
    ) -> None:
        self.trace_emitter.emit_trace_step(
            self.task,
            component=TraceComponent.RUNTIME,
            step=COMPLETION_ALIGNMENT_STEP,
            message="completion alignment authoritative assessment",
            level=TraceLevel.INFO,
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class DeferredPersistedTraceFinalize:
    """Persisted trace left open until scenario post-task observability completes."""

    trace_emitter: PersistingTaskTraceEmitter
    task: Task
    executions: tuple[AgentExecutionResult, ...]
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId

    def emit_reconciliation_phase_under_identity(
        self,
        *,
        validation_invalid: bool,
        entered_reconciliation: bool,
        phase: ReconciliationPhaseValue,
    ) -> None:
        port = TaskTraceRuntimeDiagnosticPort(
            trace_emitter=self.trace_emitter,
            task=self.task,
        )
        token = bind_active_execution_identity(
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            execution_id=self.execution_id,
        )
        try:
            port.emit_reconciliation_phase(
                run_id=self.run_id,
                validation_invalid=validation_invalid,
                entered_reconciliation=entered_reconciliation,
                phase=phase,
            )
        finally:
            reset_active_execution_identity(token)

    def emit_completion_alignment_under_identity(
        self,
        *,
        payload: CompletionAlignmentDiagV1,
    ) -> None:
        port = TaskTraceRuntimeDiagnosticPort(
            trace_emitter=self.trace_emitter,
            task=self.task,
        )
        token = bind_active_execution_identity(
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            execution_id=self.execution_id,
        )
        try:
            port.emit_completion_alignment(payload=payload)
        finally:
            reset_active_execution_identity(token)
