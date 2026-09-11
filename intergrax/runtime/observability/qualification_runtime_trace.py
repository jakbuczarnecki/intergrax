# © Artur Czarnecki. All rights reserved.

"""Emit canonical qualification runtime diagnostics into persisted trace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.execution_identity import RunId, validate_run_id
from intergrax.runtime.nexus.tracing.execution.evaluator_model_attempt import (
    EvaluatorModelAttemptDiagV1,
)
from intergrax.runtime.nexus.tracing.execution.reconciliation_phase import (
    ReconciliationPhaseDiagV1,
    ReconciliationPhaseValue,
)
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceStore
from intergrax.runtime.nexus.tracing.trace_models import (
    TraceComponent,
    TraceEvent,
    TraceLevel,
    utc_now_iso,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_trace import TaskTraceEmitter

GRAPH_QUALIFICATION_RUNTIME_TRACE_PORT_KEY = "graph_qualification_runtime_trace.v1"

EVALUATOR_MODEL_ATTEMPT_STEP = "evaluator_loop.model_attempt"
RECONCILIATION_PHASE_STEP = "completion.reconciliation_phase"

O1_SUPPORTED_TRACE_SCHEMA_IDS: tuple[str, ...] = (
    EvaluatorModelAttemptDiagV1.schema_id(),
    ReconciliationPhaseDiagV1.schema_id(),
)


class GraphQualificationRuntimeTracePort(Protocol):
    def emit_evaluator_model_attempt(
        self,
        *,
        run_id: RunId,
        node_id: str,
        attempt_index: int,
        max_iterations: int,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class TaskTraceQualificationRuntimePort:
    """Graph-scoped port backed by the active ``TaskTraceEmitter``."""

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


def resolve_graph_qualification_runtime_trace_port(
    task: Task,
) -> GraphQualificationRuntimeTracePort | None:
    port = task.metadata.get(GRAPH_QUALIFICATION_RUNTIME_TRACE_PORT_KEY)
    if port is None:
        return None
    return port  # type: ignore[return-value]


def append_reconciliation_phase_to_trace_store(
    store: RunTraceStore,
    *,
    run_id: str,
    tenant_id: str,
    attempt_index: int,
    validation_invalid: bool,
    entered_reconciliation: bool,
    phase: ReconciliationPhaseValue,
    seq: int,
) -> TraceEvent:
    """Append one reconciliation phase row to an already-finalized run trace."""
    payload = ReconciliationPhaseDiagV1(
        run_id=run_id,
        attempt_index=attempt_index,
        validation_invalid=validation_invalid,
        entered_reconciliation=entered_reconciliation,
        phase=phase,
    )
    event = TraceEvent(
        event_id=TraceEvent.new_id(),
        run_id=run_id,
        seq=seq,
        ts_utc=utc_now_iso(),
        level=TraceLevel.INFO,
        component=TraceComponent.RUNTIME,
        step=RECONCILIATION_PHASE_STEP,
        message="completion reconciliation phase",
        payload=payload,
        tags={"tenant_id": tenant_id},
    )
    store.append_event(event)
    return event


def next_trace_seq_for_run(store: RunTraceStore, run_id: str, tenant_id: str) -> int:
    persisted = store.read_run(run_id, tenant_id)
    if not persisted.events:
        return 1
    max_seq = 0
    for item in persisted.events:
        if isinstance(item, dict):
            seq_value = item.get("seq")
        else:
            seq_value = getattr(item, "seq", None)
        if isinstance(seq_value, int) and seq_value > max_seq:
            max_seq = seq_value
    return max_seq + 1
