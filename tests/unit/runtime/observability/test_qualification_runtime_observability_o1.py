# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-O1 canonical qualification runtime observability tests."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    peek_active_execution_id,
    require_active_execution_id,
)
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    peek_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.nexus.execution.evaluator_loop_metadata import tag_node_evaluator_loop
from intergrax.runtime.nexus.execution.evaluator_loop_spec import EvaluatorLoopSpec
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.tracing.execution.evaluator_model_attempt import (
    EvaluatorModelAttemptDiagV1,
)
from intergrax.runtime.nexus.tracing.execution.reconciliation_phase import (
    ReconciliationPhaseDiagV1,
    ReconciliationPhaseValue,
)
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import RunMetadata, RunStats
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.observability.qualification_runtime_trace import (
    GRAPH_QUALIFICATION_RUNTIME_TRACE_PORT_KEY,
    O1_SUPPORTED_TRACE_SCHEMA_IDS,
    TaskTraceQualificationRuntimePort,
    append_reconciliation_phase_to_trace_store,
    next_trace_seq_for_run,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from platform_proofs.scenarios.ai_incident_investigation.application.completion_transition import (
    PreReconciliationValidationError,
    enforce_pre_reconciliation_validation_clean_transition,
)
from testing_support.decision_e2e.local_qualification_session.attempt_evidence import (
    assess_third_model_pass,
    extract_attempt_observations,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
    RECONCILIATION_PHASE_TRACE_SCHEMA,
)
from testing_support.decision_e2e.local_qualification_session.reconciliation_leak import (
    assess_reconciliation_leak,
    extract_reconciliation_phase_observations,
)
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _finalize_trace_run(store: InMemoryRunTraceStore, *, run_id: str, tenant_id: str) -> None:
    store.finalize_run(
        run_id,
        RunMetadata(
            run_id=run_id,
            tenant_id=tenant_id,
            user_id="user-1",
            session_id="session-1",
            started_at_utc="2026-06-07T10:00:00+00:00",
            stats=RunStats(duration_ms=1, llm_usage={}),
        ),
    )


def _event_dicts(store: InMemoryRunTraceStore, *, run_id: str, tenant_id: str) -> tuple[dict[str, object], ...]:
    _finalize_trace_run(store, run_id=run_id, tenant_id=tenant_id)
    persisted = store.read_run(run_id, tenant_id)
    return tuple(asdict(item) for item in persisted.events)


def test_o1_schema_capability_marker() -> None:
    assert EvaluatorModelAttemptDiagV1.schema_id() in O1_SUPPORTED_TRACE_SCHEMA_IDS
    assert ReconciliationPhaseDiagV1.schema_id() in O1_SUPPORTED_TRACE_SCHEMA_IDS


def test_evaluator_model_attempt_serialization_is_stable() -> None:
    payload = EvaluatorModelAttemptDiagV1(
        run_id="run-a",
        node_id="node_investigator",
        attempt_index=1,
        max_iterations=2,
    )
    assert payload.to_dict() == {
        "run_id": "run-a",
        "node_id": "node_investigator",
        "attempt_index": 1,
        "max_iterations": 2,
    }
    event = payload.to_dict()
    again = EvaluatorModelAttemptDiagV1(
        run_id="run-a",
        node_id="node_investigator",
        attempt_index=1,
        max_iterations=2,
    )
    assert again.to_dict() == event


def test_qi1_parses_persisted_model_attempt_shape() -> None:
    observations = extract_attempt_observations(
        (
            {
                "payload_schema_id": CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
                "payload": {
                    "run_id": "run-1",
                    "node_id": "node_investigator",
                    "attempt_index": 2,
                },
            },
        )
    )
    assessment = assess_third_model_pass(observations, max_valid_attempt_index=1)
    assert assessment.outcome.value == "fail"


def test_pre_reconciliation_rejection_has_no_entered_event() -> None:
    with pytest.raises(PreReconciliationValidationError):
        enforce_pre_reconciliation_validation_clean_transition(
            validation_valid=False,
            validation_errors=("validation_rule: rejected",),
            revision_budget_remaining=0,
            completion_mode="unresolved",
            has_supported_diagnosis=False,
        )
    store = InMemoryRunTraceStore()
    append_reconciliation_phase_to_trace_store(
        store,
        run_id="run-1",
        tenant_id="tenant-1",
        attempt_index=0,
        validation_invalid=True,
        entered_reconciliation=False,
        phase=ReconciliationPhaseValue.FAILED,
        seq=1,
    )
    events = _event_dicts(store, run_id="run-1", tenant_id="tenant-1")
    observations = extract_reconciliation_phase_observations(events)
    assessment = assess_reconciliation_leak(observations)
    assert assessment.outcome.value == "pass"
    assert all(not item.entered_reconciliation for item in observations)


def test_reconciliation_entered_before_failure_is_detectable() -> None:
    store = InMemoryRunTraceStore()
    append_reconciliation_phase_to_trace_store(
        store,
        run_id="run-1",
        tenant_id="tenant-1",
        attempt_index=0,
        validation_invalid=False,
        entered_reconciliation=True,
        phase=ReconciliationPhaseValue.ENTERED,
        seq=1,
    )
    events = _event_dicts(store, run_id="run-1", tenant_id="tenant-1")
    observations = extract_reconciliation_phase_observations(events)
    leaked = extract_reconciliation_phase_observations(
        (
            {
                "payload_schema_id": RECONCILIATION_PHASE_TRACE_SCHEMA,
                "payload": {
                    "run_id": "run-1",
                    "attempt_index": 0,
                    "validation_invalid": True,
                    "entered_reconciliation": True,
                },
            },
        )
    )
    assert assess_reconciliation_leak(leaked).outcome.value == "fail"
    assert observations[0].entered_reconciliation


class _AlternatingValidationEngine(NexusValidationEngine):
    def __init__(self) -> None:
        self._calls = 0

    def validate(self, execution, *, contract, capability=None, plan_criteria=None):
        _ = execution, contract, capability, plan_criteria
        self._calls += 1
        if self._calls == 1:
            return ValidationResult(valid=False, errors=["validation_rule: retry"])
        return ValidationResult(valid=True, errors=[])


class _GraphOrchestrationDelegate:
    __slots__ = ("_executor", "_graph", "_task")

    def __init__(self, executor: GraphExecutor, graph: ExecutionGraph, task: Task) -> None:
        self._executor = executor
        self._graph = graph
        self._task = task

    async def execute(self, _request: object) -> tuple[object, ...]:
        budget_token = None
        if peek_active_execution_budget() is None:
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=create_execution_budget_ledger(None),
            )
        try:
            return await self._executor.execute(self._graph, self._task)
        finally:
            if budget_token is not None:
                reset_active_execution_budget(budget_token)


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


@pytest.mark.asyncio
@pytest.mark.gate
async def test_graph_executor_emits_attempt_indices_zero_and_one() -> None:
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(
            agent_id="incident_investigator",
            capability="incident_investigation.investigate",
            prefix="investigator",
            answer_separator=":",
        ),
    )
    executor = GraphExecutor(
        registry,
        engine=AgentEngine(registry),
        validation_engine=_AlternatingValidationEngine(),
    )
    node = ExecutionNode(
        node_id="node_incident_investigator",
        agent_id="incident_investigator",
        capability="incident_investigation.investigate",
    )
    tag_node_evaluator_loop(
        node,
        EvaluatorLoopSpec(max_iterations=2, revise_node_id="node_incident_investigator"),
    )
    graph = ExecutionGraph(
        graph_id="graph_o1_attempts",
        task_id="task_o1_attempts",
        nodes=[node],
    )
    task = Task(
        tenant_id="scenario-tenant",
        user_id="u1",
        message="investigate",
        context=TaskContext(capability="incident_investigation.investigate"),
    )
    root = _root_identity()
    trace_emitter = TaskTraceEmitter(run_id=root.run_id, attempt_id=root.attempt_id)
    task.metadata[GRAPH_QUALIFICATION_RUNTIME_TRACE_PORT_KEY] = TaskTraceQualificationRuntimePort(
        trace_emitter=trace_emitter,
        task=task,
    )
    boundary = ExecutionBoundary(
        _GraphOrchestrationDelegate(executor, graph, task),
        identity=root,
        authority=ParentExecutionAuthority.unknown(),
    )
    await boundary.execute(None)

    assert graph.node_by_id("node_incident_investigator").status is ExecutionNodeStatus.COMPLETED
    persisted = tuple(event.to_dict() for event in trace_emitter.events)
    attempt_events = [item for item in persisted if item.get("payload_schema_id") == CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA]
    indices = sorted(
        int(item["payload"]["attempt_index"])
        for item in attempt_events
        if isinstance(item.get("payload"), dict)
    )
    assert indices == [0, 1]
    assert peek_active_execution_id() is None


def test_next_trace_seq_increments() -> None:
    store = InMemoryRunTraceStore()
    append_reconciliation_phase_to_trace_store(
        store,
        run_id="run-seq",
        tenant_id="tenant-1",
        attempt_index=0,
        validation_invalid=False,
        entered_reconciliation=True,
        phase=ReconciliationPhaseValue.ENTERED,
        seq=3,
    )
    _finalize_trace_run(store, run_id="run-seq", tenant_id="tenant-1")
    assert next_trace_seq_for_run(store, "run-seq", "tenant-1") == 4
