# © Artur Czarnecki. All rights reserved.

"""Graph failure propagation — preserve failed node execution and typed validation errors."""

from __future__ import annotations

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
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
from intergrax.runtime.nexus.orchestration.graph_runner import graph_failure_validation_errors
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.asyncio]


class _AlwaysFailValidationEngine(NexusValidationEngine):
    def validate(self, execution, *, contract, capability=None, plan_criteria=None):
        _ = execution, contract, capability, plan_criteria
        return ValidationResult(valid=False, errors=["validation_rule: scenario_rejected"])


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


async def _run_graph(
    executor: GraphExecutor,
    graph: ExecutionGraph,
    task: Task,
    root: ExecutionIdentityBinding,
) -> tuple[object, ...]:
    boundary = ExecutionBoundary(
        _GraphOrchestrationDelegate(executor, graph, task),
        identity=root,
        authority=ParentExecutionAuthority.unknown(),
    )
    return await boundary.execute(None)


@pytest.mark.asyncio
async def test_failed_graph_node_execution_is_retained_for_finalization() -> None:
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
        validation_engine=_AlwaysFailValidationEngine(),
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=0)),
    )
    node = ExecutionNode(
        node_id="node_incident_investigator",
        agent_id="incident_investigator",
        capability="incident_investigation.investigate",
    )
    tag_node_evaluator_loop(
        node,
        EvaluatorLoopSpec(max_iterations=1, revise_node_id="node_incident_investigator"),
    )
    graph = ExecutionGraph(
        graph_id="graph_failure_propagation",
        task_id="task_failure_propagation",
        nodes=[node],
    )
    task = Task(
        tenant_id="scenario-tenant",
        user_id="u1",
        message="investigate",
        context=TaskContext(capability="incident_investigation.investigate"),
    )
    root = _root_identity()

    executions, _retries, completed_graph, cancelled = await _run_graph(
        executor,
        graph,
        task,
        root,
    )

    assert cancelled is False
    assert completed_graph.node_by_id("node_incident_investigator").status is ExecutionNodeStatus.FAILED
    assert len(executions) == 1
    assert executions[0].status is AgentExecutionStatus.COMPLETED
    assert (executions[0].summary or "").strip() != ""
    assert peek_active_execution_id() is None


def test_graph_failure_validation_errors_preserves_node_validation_errors() -> None:
    node = ExecutionNode(
        node_id="node_incident_investigator",
        agent_id="incident_investigator",
        capability="incident_investigation.investigate",
        status=ExecutionNodeStatus.FAILED,
    )
    node.metadata["node_validation_errors"] = [
        "unsupported_inference:missing_comparison_evidence",
    ]
    graph = ExecutionGraph(
        graph_id="graph_failure_validation",
        task_id="task_failure_validation",
        nodes=[node],
    )

    errors = graph_failure_validation_errors(graph, ["node_incident_investigator"])

    assert errors[0] == "graph node failed: ['node_incident_investigator']"
    assert "unsupported_inference:missing_comparison_evidence" in errors
