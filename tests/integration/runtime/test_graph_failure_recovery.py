# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    require_active_execution_id,
    reset_active_execution_identity,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.long_running.checkpoint_builder import (
    apply_runtime_checkpoint_to_task,
    build_runtime_checkpoint,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep, TaskPlanner
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent


def _bind_nexus_upstream_context(
    *,
    run_id,
    attempt_id,
    execution_id,
):
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.unknown(),
    )
    budget_token = bind_root_execution_budget(
        execution_id=execution_id,
        ledger=create_execution_budget_ledger(None),
    )
    return identity_token, authority_token, budget_token


def _reset_nexus_upstream_context(
    *,
    identity_token,
    authority_token,
    budget_token,
) -> None:
    reset_active_execution_budget(budget_token)
    reset_active_execution_authority(authority_token)
    reset_active_execution_identity(identity_token)


class _FlakyValidationEngine(NexusValidationEngine):
    """Fails agent_b validation for the first N attempts (simulates graph node failure)."""

    def __init__(self, *, fail_agent_b_times: int = 1) -> None:
        super().__init__()
        self._fail_agent_b_times = fail_agent_b_times
        self._agent_b_failures = 0

    def validate(
        self,
        execution,
        *,
        contract,
        capability=None,
        plan_criteria=None,
    ) -> ValidationResult:
        if (
            execution.agent_id == "agent_b"
            and self._agent_b_failures < self._fail_agent_b_times
        ):
            self._agent_b_failures += 1
            return ValidationResult(valid=False, errors=["simulated node failure"])
        return super().validate(
            execution,
            contract=contract,
            capability=capability,
            plan_criteria=plan_criteria,
        )


class _TwoStepPlanner(TaskPlanner):
    def plan(self, task: Task, registry: AgentRegistry) -> NexusPlan:
        _ = registry
        return NexusPlan(
            task_id=task.task_id,
            classification=task.classification or "long_running",
            steps=[
                PlanStep(
                    step_id="step_1",
                    agent_id="agent_a",
                    capability="graph.recovery",
                    description="first step",
                ),
                PlanStep(
                    step_id="step_2",
                    agent_id="agent_b",
                    capability="graph.recovery",
                    description="second step",
                    depends_on=["step_1"],
                ),
            ],
            validation_criteria=["non_empty_summary"],
        )


def _build_graph(task_id: str) -> ExecutionGraph:
    return ExecutionGraph(
        graph_id="graph_recovery_1",
        task_id=task_id,
        nodes=[
            ExecutionNode(node_id="n1", agent_id="agent_a", capability="graph.recovery"),
            ExecutionNode(
                node_id="n2",
                agent_id="agent_b",
                capability="graph.recovery",
                depends_on=["n1"],
            ),
        ],
    )


async def _execute_graph(executor, graph, task, *, run_id, attempt_id):
    execution_id = mint_execution_id()
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.unknown(),
    )
    budget_token = bind_root_execution_budget(
        execution_id=execution_id,
        ledger=create_execution_budget_ledger(None),
    )
    try:
        return await executor.execute(graph, task)
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_graph_executor_skips_completed_nodes_on_resume():
    UaepPipelineStubAgent.run_count = 0
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(agent_id="agent_a", capability="graph.recovery", prefix="A")
    )
    registry.register(
        UaepPipelineStubAgent(agent_id="agent_b", capability="graph.recovery", prefix="B")
    )

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="recover graph",
        context=TaskContext(capability="graph.recovery"),
    )
    graph = _build_graph(task.task_id)
    validation = _FlakyValidationEngine(fail_agent_b_times=1)
    executor = GraphExecutor(
        registry,
        validation_engine=validation,
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=0)),
    )

    run_id = mint_run_id()
    attempt_id = mint_attempt_id()

    executions, _, graph, _ = await _execute_graph(
        executor,
        graph,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    assert len(executions) == 1
    assert graph.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED
    assert graph.node_by_id("n2").status == ExecutionNodeStatus.FAILED
    assert UaepPipelineStubAgent.run_count == 2

    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    try:
        runtime = build_runtime_checkpoint(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            graph=graph,
            last_execution=executions[-1],
        )
    finally:
        reset_active_execution_identity(token)
    apply_runtime_checkpoint_to_task(task, runtime)

    resumed_graph = _build_graph(task.task_id)
    executions, _, graph, _ = await _execute_graph(
        executor,
        resumed_graph,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
    )

    assert len(executions) == 2
    assert executions[0].summary == "A: recover graph"
    assert executions[1].agent_id == "agent_b"
    assert "B: recover graph" in executions[1].summary
    assert graph.node_by_id("n1").status == ExecutionNodeStatus.SKIPPED
    assert graph.node_by_id("n2").status == ExecutionNodeStatus.COMPLETED
    assert UaepPipelineStubAgent.run_count == 3


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_graph_failure_resume(tmp_path):
    UaepPipelineStubAgent.run_count = 0
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(agent_id="agent_a", capability="graph.recovery", prefix="A")
    )
    registry.register(
        UaepPipelineStubAgent(agent_id="agent_b", capability="graph.recovery", prefix="B")
    )
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    validation = _FlakyValidationEngine(fail_agent_b_times=1)

    loop = NexusLoop(
        registry,
        planner=_TwoStepPlanner(),
        checkpoint_store=checkpoint_store,
        validation_engine=validation,
        retry_policy=RetryPolicy(max_retries=0),
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    identity_token, authority_token, budget_token = _bind_nexus_upstream_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        failed = await loop.handle_task(
            Task(
                tenant_id="t1",
                user_id="u1",
                message="multi-step recovery",
                context=TaskContext(capability="graph.recovery"),
                options=TaskExecutionOptions(
                    long_running=TaskLongRunningOptions(enabled=True),
                ),
            ),
            run_id=run_id,
            attempt_id=attempt_id,
        )
    finally:
        _reset_nexus_upstream_context(
            identity_token=identity_token,
            authority_token=authority_token,
            budget_token=budget_token,
        )
    assert failed.state == TaskState.FAILED
    token = failed.summary.resume_token
    assert token
    assert checkpoint_store.get_latest(failed.task_id, "t1") is not None

    identity_token, authority_token, budget_token = _bind_nexus_upstream_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    try:
        completed = await loop.handle_task(
            Task(
                tenant_id="t1",
                user_id="u1",
                message="multi-step recovery",
                context=TaskContext(capability="graph.recovery"),
                task_id=failed.task_id,
                options=TaskExecutionOptions(
                    long_running=TaskLongRunningOptions(
                        enabled=True,
                        resume_token=token,
                    ),
                ),
            ),
            run_id=run_id,
            attempt_id=attempt_id,
        )
    finally:
        _reset_nexus_upstream_context(
            identity_token=identity_token,
            authority_token=authority_token,
            budget_token=budget_token,
        )

    assert completed.state == TaskState.COMPLETED
    assert UaepPipelineStubAgent.run_count == 3
