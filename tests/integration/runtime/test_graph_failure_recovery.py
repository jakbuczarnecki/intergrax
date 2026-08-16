# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.long_running.checkpoint_builder import build_runtime_checkpoint
from intergrax.runtime.long_running.runtime_checkpoint import attach_runtime_checkpoint_to_metadata
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

    executions, _, graph, _ = await executor.execute(graph, task)
    assert len(executions) == 1
    assert graph.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED
    assert graph.node_by_id("n2").status == ExecutionNodeStatus.FAILED
    assert UaepPipelineStubAgent.run_count == 2

    runtime = build_runtime_checkpoint(
        task,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        graph=graph,
        last_execution=executions[-1],
    )
    attach_runtime_checkpoint_to_metadata(task.metadata, runtime)
    task.sync_metadata()

    resumed_graph = _build_graph(task.task_id)
    executions, _, graph, _ = await executor.execute(resumed_graph, task)

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
    failed = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="multi-step recovery",
            context=TaskContext(capability="graph.recovery"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True),
            ),
        )
    )
    assert failed.state == TaskState.FAILED
    token = failed.summary.resume_token
    assert token
    assert checkpoint_store.get_latest(failed.task_id, "t1") is not None

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
        )
    )

    assert completed.state == TaskState.COMPLETED
    assert UaepPipelineStubAgent.run_count == 3
