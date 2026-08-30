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
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.long_running.checkpoint_builder import apply_runtime_checkpoint_to_task
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionPriorOutput,
    ExecutionTreeSnapshot,
)
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
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


class RouterInvocationRecorder:
    __slots__ = ("_router", "invocations")

    def __init__(self, router: StrategyExecutionRouter) -> None:
        self._router = router
        self.invocations: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> object:
        self.invocations.append(request)
        return await self._router.execute(request)


def _install_capability_observer(
    executor: GraphExecutor,
    engine: ObservingAgentEngine,
) -> CapabilityObservingAgentExecutor:
    capability_executor = CapabilityObservingAgentExecutor(AgentExecutor(engine))
    executor._strategy_router = StrategyExecutionRouter(  # noqa: SLF001
        agent_executor=capability_executor,
    )
    return capability_executor


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
    capability_executor: CapabilityObservingAgentExecutor


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
    capability_executor = _install_capability_observer(executor, engine)
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
        capability_executor=capability_executor,
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
        from intergrax.runtime.execution.active_execution_budget import (
            bind_root_execution_budget,
            peek_active_execution_budget,
            reset_active_execution_budget,
        )
        from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger

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


async def _run_graph(ctx: _GraphRunContext, graph: ExecutionGraph) -> tuple[object, ...]:
    boundary = ExecutionBoundary(
        _GraphOrchestrationDelegate(ctx.executor, graph, ctx.task),
        identity=ctx.root,
        authority=ParentExecutionAuthority.unknown(),
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

    capability_executor = ctx.capability_executor
    assert capability_executor.last_capabilities == frozenset({ExecutionCapability.AGENT})


async def test_child_agentic_execution_routes_through_strategy_router() -> None:
    ctx = _build_graph_run(
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )
    graph = ExecutionGraph(
        graph_id="strategy-router-child",
        task_id=ctx.task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )
    inner_router = StrategyExecutionRouter(agent_executor=AgentExecutor(ctx.engine))
    recorder = RouterInvocationRecorder(inner_router)
    ctx.executor._strategy_router = recorder  # noqa: SLF001

    await _run_graph(ctx, graph)

    assert len(recorder.invocations) == 1
    request = recorder.invocations[0]
    assert request.capabilities == frozenset({ExecutionCapability.AGENT})
    assert StrategyResolver().resolve(request) is ExecutionStrategy.AGENTIC
    assert len(ctx.engine.observations) == 1


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
    capability_executor = _install_capability_observer(executor, engine)
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
        capability_executor=capability_executor,
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
    apply_runtime_checkpoint_to_task(
        ctx.task,
        RuntimeCheckpoint(
            run_id=ctx.root.run_id,
            attempt_id=ctx.root.attempt_id,
            execution_tree=ExecutionTreeSnapshot(
                task_id=ctx.task.task_id,
                run_id=ctx.root.run_id,
                attempt_id=ctx.root.attempt_id,
                entries=[
                    ExecutionCheckpointEntry(
                        execution_id=ctx.root.execution_id,
                        parent_execution_id=None,
                        status=ExecutionCheckpointStatus.RUNNING,
                    ),
                    ExecutionCheckpointEntry(
                        execution_id=mint_execution_id(),
                        parent_execution_id=ctx.root.execution_id,
                        status=ExecutionCheckpointStatus.COMPLETED,
                        graph_node_id="n1",
                        prior_output=ExecutionPriorOutput(
                            agent_id="agent_a",
                            summary="cached",
                            status=AgentExecutionStatus.COMPLETED.value,
                            graph_node_id="n1",
                        ),
                    ),
                ],
            ),
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
    capability_executor = _install_capability_observer(executor, engine)
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
        capability_executor=capability_executor,
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
