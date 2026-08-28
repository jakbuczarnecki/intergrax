# © Artur Czarnecki. All rights reserved.

"""GraphExecutor child execution lineage proofs (UE-7B)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    peek_active_parent_execution_id,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.execution import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.agentic import AgentExecutor
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.asyncio]


@dataclass
class IdentityObservation:
    execution_id: ExecutionId
    parent_execution_id: ExecutionId | None
    run_id: RunId
    attempt_id: AttemptId


class ObservingAgentEngine(AgentEngine):
    """AgentEngine recording active execution identity during run_with_result."""

    def __init__(self, registry: AgentRegistry) -> None:
        super().__init__(registry)
        self.observations: list[IdentityObservation] = []

    async def run_with_result(self, request):
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        self.observations.append(
            IdentityObservation(
                execution_id=execution_id,
                parent_execution_id=peek_active_parent_execution_id(),
                run_id=run_id,
                attempt_id=attempt_id,
            ),
        )
        return await super().run_with_result(request)


class CapabilityObservingAgentExecutor:
    __slots__ = ("_inner", "last_capabilities")

    def __init__(self, inner: AgentExecutor) -> None:
        self._inner = inner
        self.last_capabilities: frozenset[ExecutionCapability] | None = None

    async def execute(self, request: ExecutionRequest) -> object:
        self.last_capabilities = request.capabilities
        return await self._inner.execute(request)


class _FailOnceValidation(NexusValidationEngine):
    def __init__(self, *, fail_agent: str) -> None:
        super().__init__()
        self._fail_agent = fail_agent
        self._failed: set[str] = set()

    def validate(
        self,
        execution: AgentExecutionResult,
        *,
        contract,
        capability=None,
        plan_criteria=None,
    ) -> ValidationResult:
        agent_id = contract.id
        if agent_id == self._fail_agent and agent_id not in self._failed:
            self._failed.add(agent_id)
            return ValidationResult(valid=False, errors=["simulated validation failure"])
        return super().validate(
            execution,
            contract=contract,
            capability=capability,
            plan_criteria=plan_criteria,
        )


@dataclass
class _GraphRunContext:
    registry: AgentRegistry
    engine: ObservingAgentEngine
    executor: GraphExecutor
    root: ExecutionIdentityBinding
    task: Task


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _build_graph_run(
    *,
    nodes: list[ExecutionNode],
    validation_engine: NexusValidationEngine | None = None,
    retry_engine: RetryEngine | None = None,
) -> _GraphRunContext:
    registry = AgentRegistry()
    for node in nodes:
        if node.agent_id and not registry.has(node.agent_id):
            registry.register(
                UaepPipelineStubAgent(
                    agent_id=node.agent_id,
                    capability=node.capability or "cap.shared",
                    prefix=node.agent_id,
                    answer_separator=":",
                ),
            )
    engine = ObservingAgentEngine(registry)
    executor = GraphExecutor(
        registry,
        engine=engine,
        validation_engine=validation_engine or NexusValidationEngine(),
        retry_engine=retry_engine,
    )
    capability_executor = CapabilityObservingAgentExecutor(AgentExecutor(engine))
    executor._agent_executor = capability_executor  # noqa: SLF001
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="child execution proof",
        context=TaskContext(capability=nodes[0].capability or "cap.shared"),
    )
    return _GraphRunContext(
        registry=registry,
        engine=engine,
        executor=executor,
        root=_root_identity(),
        task=task,
    )


class _GraphOrchestrationDelegate:
    __slots__ = ("_executor", "_graph", "_task")

    def __init__(
        self,
        executor: GraphExecutor,
        graph: ExecutionGraph,
        task: Task,
    ) -> None:
        self._executor = executor
        self._graph = graph
        self._task = task

    async def execute(self, _request: object) -> tuple[object, ...]:
        return await self._executor.execute(self._graph, self._task)


async def _run_graph(ctx: _GraphRunContext, graph: ExecutionGraph) -> tuple[object, ...]:
    boundary = ExecutionBoundary(
        _GraphOrchestrationDelegate(ctx.executor, graph, ctx.task),
        identity=ctx.root,
    )
    return await boundary.execute(None)


async def test_single_node_runs_as_child_of_orchestration_execution() -> None:
    ctx = _build_graph_run(
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )
    graph = ExecutionGraph(
        graph_id="child-single",
        task_id=ctx.task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )

    await _run_graph(ctx, graph)

    assert len(ctx.engine.observations) == 1
    observation = ctx.engine.observations[0]
    assert observation.parent_execution_id == ctx.root.execution_id
    assert observation.run_id == ctx.root.run_id
    assert observation.attempt_id == ctx.root.attempt_id
    assert observation.execution_id != ctx.root.execution_id
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


async def test_agent_executor_receives_agent_capability() -> None:
    ctx = _build_graph_run(
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )
    graph = ExecutionGraph(
        graph_id="agent-cap",
        task_id=ctx.task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )

    await _run_graph(ctx, graph)

    capability_executor = ctx.executor._agent_executor  # noqa: SLF001
    assert capability_executor.last_capabilities == frozenset({ExecutionCapability.AGENT})


async def test_agent_engine_resolves_selected_request_agent_id() -> None:
    ctx = _build_graph_run(
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )
    graph = ExecutionGraph(
        graph_id="agent-id",
        task_id=ctx.task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )

    executions, _, graph_out, _ = await _run_graph(ctx, graph)

    assert executions[0].agent_id == "agent_a"
    assert graph_out.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED


async def test_local_retry_preserves_child_execution_id() -> None:
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(
            agent_id="agent_a",
            capability="cap.shared",
            prefix="agent_a",
            answer_separator=":",
        ),
    )
    registry.register(
        UaepPipelineStubAgent(
            agent_id="agent_b",
            capability="cap.shared",
            prefix="agent_b",
            answer_separator=":",
        ),
    )
    engine = ObservingAgentEngine(registry)
    validation = _FailOnceValidation(fail_agent="agent_a")
    executor = GraphExecutor(
        registry,
        engine=engine,
        validation_engine=validation,
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=1)),
    )
    executor._agent_executor = CapabilityObservingAgentExecutor(AgentExecutor(engine))
    ctx = _GraphRunContext(
        registry=registry,
        engine=engine,
        executor=executor,
        root=_root_identity(),
        task=Task(
            tenant_id="t1",
            user_id="u1",
            message="child execution proof",
            context=TaskContext(capability="cap.shared"),
        ),
    )
    graph = ExecutionGraph(
        graph_id="retry-child",
        task_id=ctx.task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )

    executions, retries, graph_out, _ = await _run_graph(ctx, graph)

    assert len(retries) == 1
    assert len(ctx.engine.observations) == 2
    first, second = ctx.engine.observations
    assert first.execution_id == second.execution_id
    assert first.parent_execution_id == ctx.root.execution_id
    assert second.parent_execution_id == ctx.root.execution_id
    assert first.run_id == ctx.root.run_id == second.run_id
    assert first.attempt_id == ctx.root.attempt_id == second.attempt_id
    assert graph_out.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED
    assert executions[-1].agent_id == "agent_b"


async def test_sequential_nodes_receive_distinct_child_execution_ids() -> None:
    ctx = _build_graph_run(
        nodes=[
            ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared"),
            ExecutionNode(
                node_id="n2",
                agent_id="agent_b",
                capability="cap.shared",
                depends_on=["n1"],
            ),
        ],
    )
    graph = ExecutionGraph(
        graph_id="sequential-child",
        task_id=ctx.task.task_id,
        nodes=[
            ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared"),
            ExecutionNode(
                node_id="n2",
                agent_id="agent_b",
                capability="cap.shared",
                depends_on=["n1"],
            ),
        ],
    )

    await _run_graph(ctx, graph)

    assert len(ctx.engine.observations) == 2
    first, second = ctx.engine.observations
    assert first.execution_id != second.execution_id
    assert first.parent_execution_id == ctx.root.execution_id
    assert second.parent_execution_id == ctx.root.execution_id


async def test_parallel_nodes_receive_unique_child_execution_ids() -> None:
    ctx = _build_graph_run(
        nodes=[
            ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared"),
            ExecutionNode(node_id="n2", agent_id="agent_b", capability="cap.shared"),
            ExecutionNode(node_id="n3", agent_id="agent_c", capability="cap.shared"),
        ],
    )
    graph = ExecutionGraph(
        graph_id="parallel-child",
        task_id=ctx.task.task_id,
        nodes=[
            ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared"),
            ExecutionNode(node_id="n2", agent_id="agent_b", capability="cap.shared"),
            ExecutionNode(node_id="n3", agent_id="agent_c", capability="cap.shared"),
        ],
    )

    await _run_graph(ctx, graph)

    assert len(ctx.engine.observations) == 3
    execution_ids = {observation.execution_id for observation in ctx.engine.observations}
    assert len(execution_ids) == 3
    for observation in ctx.engine.observations:
        assert observation.parent_execution_id == ctx.root.execution_id
        assert observation.run_id == ctx.root.run_id
        assert observation.attempt_id == ctx.root.attempt_id


async def test_checkpoint_skip_does_not_mint_child_execution() -> None:
    ctx = _build_graph_run(
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )
    graph = ExecutionGraph(
        graph_id="skip-child",
        task_id=ctx.task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="agent_a",
                capability="cap.shared",
                status=ExecutionNodeStatus.COMPLETED,
            ),
        ],
    )
    from intergrax.runtime.long_running.runtime_checkpoint import (
        RuntimeCheckpointExecutionState,
        attach_runtime_checkpoint_to_metadata,
    )

    attach_runtime_checkpoint_to_metadata(
        ctx.task.metadata,
        RuntimeCheckpointExecutionState(
            node_states={"n1": ExecutionNodeStatus.COMPLETED.value},
            prior_node_outputs={
                "n1": {
                    "agent_id": "agent_a",
                    "summary": "cached",
                    "status": AgentExecutionStatus.COMPLETED.value,
                },
            },
        ),
    )

    await _run_graph(ctx, graph)

    assert ctx.engine.observations == []


async def test_no_execution_identity_metadata_added() -> None:
    registry = AgentRegistry()
    tracking_agent = UaepPipelineStubAgent(
        agent_id="agent_a",
        capability="cap.shared",
        prefix="agent_a",
        answer_separator=":",
        track_request_metadata=True,
    )
    registry.register(tracking_agent)
    engine = ObservingAgentEngine(registry)
    executor = GraphExecutor(registry, engine=engine)
    executor._agent_executor = CapabilityObservingAgentExecutor(AgentExecutor(engine))
    ctx = _GraphRunContext(
        registry=registry,
        engine=engine,
        executor=executor,
        root=_root_identity(),
        task=Task(
            tenant_id="t1",
            user_id="u1",
            message="child execution proof",
            context=TaskContext(capability="cap.shared"),
        ),
    )
    graph = ExecutionGraph(
        graph_id="metadata-clean",
        task_id=ctx.task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )

    await _run_graph(ctx, graph)

    assert tracking_agent.last_request is not None
    assert "execution_id" not in tracking_agent.last_metadata
    assert "parent_execution_id" not in tracking_agent.last_metadata
