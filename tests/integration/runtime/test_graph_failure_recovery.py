# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.long_running.checkpoint_builder import build_runtime_checkpoint
from intergrax.runtime.long_running.runtime_checkpoint import attach_runtime_checkpoint_to_metadata
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep, TaskPlanner
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _AnswerPipeline(RuntimePipeline):
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        answer = f"{self._prefix}: {state.request.message}"
        state.raw_answer = answer
        state.runtime_answer = RuntimeAnswer(run_id=state.run_id, answer=answer)
        return state.runtime_answer


class _GraphRecoveryAgent(Agent):
    run_count = 0

    def __init__(self, *, agent_id: str, prefix: str) -> None:
        self._agent_id = agent_id
        self._prefix = prefix

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="graph recovery stub",
            capabilities=["graph.recovery"],
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        if task_context.capability == "graph.recovery":
            return CapabilityMatchResult(
                matched=True,
                agent_id=self._agent_id,
                matched_capabilities=["graph.recovery"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        _GraphRecoveryAgent.run_count += 1
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text=f"{self._prefix}: {request.message}"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        config.pipeline = _AnswerPipeline(self._prefix)
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


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
    _GraphRecoveryAgent.run_count = 0
    registry = AgentRegistry()
    registry.register(_GraphRecoveryAgent(agent_id="agent_a", prefix="A"))
    registry.register(_GraphRecoveryAgent(agent_id="agent_b", prefix="B"))

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
    assert _GraphRecoveryAgent.run_count == 2

    runtime = build_runtime_checkpoint(task, graph=graph, last_execution=executions[-1])
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
    assert _GraphRecoveryAgent.run_count == 3


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_graph_failure_resume(tmp_path):
    _GraphRecoveryAgent.run_count = 0
    registry = AgentRegistry()
    registry.register(_GraphRecoveryAgent(agent_id="agent_a", prefix="A"))
    registry.register(_GraphRecoveryAgent(agent_id="agent_b", prefix="B"))
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
    assert _GraphRecoveryAgent.run_count == 3
