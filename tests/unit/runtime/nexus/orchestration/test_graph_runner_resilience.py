# © Artur Czarnecki. All rights reserved.

"""Tests for graph_runner resilience policy wiring (FLOW-MAINT-01, FLOW-MAINT-05)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.resilience_policy import ResiliencePolicy
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.nexus.orchestration.graph_runner import NexusGraphRunner
from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.nexus.response.final_response_composer import FinalResponseComposer
from intergrax.runtime.execution.attempt_lifecycle import AttemptLifecycleService, InMemoryAttemptLifecycleStore
from intergrax.runtime.nexus.retry.retry_engine import _resilience_policy_from_task
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import TaskTraceEmitter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resilience_policy_disallows_partial_result() -> None:
    task = Task(
        task_id="t1",
        tenant_id="tenant",
        user_id="user",
        message="hello",
        metadata={
            "resilience_policy.v1": ResiliencePolicy(allow_partial_result=False).model_dump(),
        },
    )
    policy = _resilience_policy_from_task(task)
    assert policy is not None
    assert policy.allow_partial_result is False


def test_graph_runner_module_exports_runner() -> None:
    assert NexusGraphRunner is not None


def _build_runner(
    *,
    executions: list[AgentExecutionResult],
    graph: ExecutionGraph,
) -> NexusGraphRunner:
    registry = MagicMock()
    agent = MagicMock()
    agent.get_contract.return_value = AgentContract(
        id="agent_a",
        name="agent_a",
        description="graph runner resilience stub",
        capabilities=["test.cap"],
    )
    registry.get.return_value = agent

    graph_executor = MagicMock()
    graph_executor.execute = AsyncMock(return_value=(executions, [], graph, False))
    graph_executor.set_retry_policy = MagicMock()

    validation_engine = MagicMock()
    validation_engine.validate.return_value = ValidationResult(valid=True)

    events = MagicMock()
    events.publish = AsyncMock()

    return NexusGraphRunner(
        registry=registry,
        graph_executor=graph_executor,
        validation_engine=validation_engine,
        composer=FinalResponseComposer(),
        hitl=MagicMock(),
        events=events,
        finish_task=AsyncMock(),
        finalize_trace=AsyncMock(),
        maybe_checkpoint=AsyncMock(),
        attempt_lifecycle=AttemptLifecycleService(InMemoryAttemptLifecycleStore()),
    )


def _partial_multi_node_graph(task_id: str) -> ExecutionGraph:
    return ExecutionGraph(
        graph_id="graph_partial",
        task_id=task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="agent_a",
                capability="test.cap",
                status=ExecutionNodeStatus.COMPLETED,
            ),
            ExecutionNode(
                node_id="n2",
                agent_id="agent_a",
                capability="test.cap",
                status=ExecutionNodeStatus.COMPLETED,
            ),
        ],
    )


def _mixed_executions() -> list[AgentExecutionResult]:
    return [
        AgentExecutionResult(
            agent_id="agent_a",
            run_id="run_1",
            status=AgentExecutionStatus.COMPLETED,
            summary="done",
        ),
        AgentExecutionResult(
            agent_id="agent_a",
            run_id="run_2",
            status=AgentExecutionStatus.PARTIAL,
            summary="partial",
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allow_partial_result", "expected_state"),
    [
        (False, TaskState.FAILED),
        (True, TaskState.PARTIALLY_COMPLETED),
    ],
)
async def test_graph_runner_honors_allow_partial_result_lifecycle(
    allow_partial_result: bool,
    expected_state: TaskState,
) -> None:
    task = Task(
        task_id="task_partial_policy",
        tenant_id="tenant",
        user_id="user",
        message="hello",
        state=TaskState.RUNNING,
        metadata={
            "resilience_policy.v1": ResiliencePolicy(
                allow_partial_result=allow_partial_result,
            ).model_dump(),
        },
    )
    graph = _partial_multi_node_graph(task.task_id)
    runner = _build_runner(executions=_mixed_executions(), graph=graph)
    lifecycle = TaskLifecycle()
    trace_emitter = MagicMock(spec=TaskTraceEmitter)
    plan = NexusPlan(task_id=task.task_id, classification="test")

    await runner.run(
        task,
        plan=plan,
        graph=graph,
        lifecycle=lifecycle,
        trace_emitter=trace_emitter,
    )

    assert task.state is expected_state
