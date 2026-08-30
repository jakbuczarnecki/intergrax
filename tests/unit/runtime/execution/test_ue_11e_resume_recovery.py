# © Artur Czarnecki. All rights reserved.

"""UE-11E — checkpoint resume execution tree continuity qualification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    peek_active_execution_budget,
    require_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.governance.active_execution_authority import peek_active_execution_authority
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.consumption import consume_llm_call
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.long_running.checkpoint_builder import (
    apply_runtime_checkpoint_to_task,
    build_runtime_checkpoint,
    prepare_task_for_checkpoint_resume,
    resolve_task_runtime_checkpoint,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointStatus,
    ExecutionTreeRecorder,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_NODE_A = "n_a"
_NODE_B = "n_b"
_NODE_C = "n_c"
_NODE_D = "n_d"
_AGENT_A = "agent_a"
_AGENT_B = "agent_b"
_AGENT_C = "agent_c"
_AGENT_D = "agent_d"
_CAPABILITY = "cap.shared"
_AUTHORITY = ParentExecutionAuthority.scoped(("capability:read",))


@dataclass(frozen=True, slots=True)
class _NodeInvocationCounts:
    agent_a: int
    agent_b: int
    agent_c: int
    agent_d: int


class _CountingAgentEngine(AgentEngine):
    __slots__ = ("_counts",)

    def __init__(self, registry: AgentRegistry) -> None:
        super().__init__(registry)
        self._counts: dict[str, int] = {
            _AGENT_A: 0,
            _AGENT_B: 0,
            _AGENT_C: 0,
            _AGENT_D: 0,
        }

    async def run_with_result(self, request):
        agent_id = request.agent_id
        if agent_id in self._counts:
            self._counts[agent_id] += 1
        execution_id = require_active_execution_id()
        consume_llm_call()
        assert require_active_execution_budget().execution_id == execution_id
        return await super().run_with_result(request)

    def snapshot_counts(self) -> _NodeInvocationCounts:
        return _NodeInvocationCounts(
            agent_a=self._counts[_AGENT_A],
            agent_b=self._counts[_AGENT_B],
            agent_c=self._counts[_AGENT_C],
            agent_d=self._counts[_AGENT_D],
        )


class _GatedCountingAgentEngine(_CountingAgentEngine):
    __slots__ = ("_gate_open")

    def __init__(self, registry: AgentRegistry, *, gate_open: bool) -> None:
        super().__init__(registry)
        self._gate_open = gate_open

    async def run_with_result(self, request):
        agent_id = request.agent_id
        if not self._gate_open and agent_id == _AGENT_C:
            if agent_id in self._counts:
                self._counts[agent_id] += 1
            execution_id = require_active_execution_id()
            consume_llm_call()
            assert require_active_execution_budget().execution_id == execution_id
            run_id, _ = require_active_execution_identity()
            return AgentExecutionResult(
                agent_id=agent_id,
                run_id=run_id,
                status=AgentExecutionStatus.FAILED,
                summary="controlled interruption gate",
                errors=["controlled interruption gate"],
            )
        return await super().run_with_result(request)


def _sequential_graph(task_id: str) -> ExecutionGraph:
    return ExecutionGraph(
        graph_id="ue-11e-resume-tree",
        task_id=task_id,
        nodes=[
            ExecutionNode(node_id=_NODE_A, agent_id=_AGENT_A, capability=_CAPABILITY),
            ExecutionNode(
                node_id=_NODE_B,
                agent_id=_AGENT_B,
                capability=_CAPABILITY,
                depends_on=[_NODE_A],
            ),
            ExecutionNode(
                node_id=_NODE_C,
                agent_id=_AGENT_C,
                capability=_CAPABILITY,
                depends_on=[_NODE_B],
            ),
            ExecutionNode(
                node_id=_NODE_D,
                agent_id=_AGENT_D,
                capability=_CAPABILITY,
                depends_on=[_NODE_C],
            ),
        ],
    )


def _register_resume_agents(registry: AgentRegistry) -> None:
    for agent_id in (_AGENT_A, _AGENT_B, _AGENT_C, _AGENT_D):
        registry.register(
            UaepPipelineStubAgent(
                agent_id=agent_id,
                capability=_CAPABILITY,
                prefix=agent_id,
                answer_separator=":",
            ),
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
        budget_token = bind_root_execution_budget(
            execution_id=require_active_execution_id(),
            ledger=create_execution_budget_ledger(RunBudget(max_llm_calls=20)),
        )
        try:
            return await self._executor.execute(self._graph, self._task)
        finally:
            reset_active_execution_budget(budget_token)


async def _run_graph(
    *,
    executor: GraphExecutor,
    task: Task,
    graph: ExecutionGraph,
    run_id: RunId,
    attempt_id: AttemptId,
    root_execution_id: ExecutionId,
) -> tuple[object, ...]:
    boundary = ExecutionBoundary(
        _GraphOrchestrationDelegate(executor, graph, task),
        identity=ExecutionIdentityBinding(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=root_execution_id,
        ),
        authority=_AUTHORITY,
    )
    return await boundary.execute(None)


def _execution_id_for_node(
    runtime_checkpoint,
    node_id: str,
) -> ExecutionId | None:
    entry = runtime_checkpoint.execution_tree.entry_by_graph_node_id(node_id)
    if entry is None:
        return None
    return entry.execution_id


async def test_ue_11e_resume_execution_tree_continuity(tmp_path: Path) -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_before = mint_execution_id()

    registry = AgentRegistry()
    _register_resume_agents(registry)
    engine = _GatedCountingAgentEngine(registry, gate_open=False)
    interrupt_executor = GraphExecutor(
        registry,
        engine=engine,
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=0)),
    )

    task_a = Task(
        task_id=task_id,
        tenant_id="t1",
        user_id="u1",
        message="ue-11e resume interrupt",
        context=TaskContext(capability=_CAPABILITY),
    )
    graph = _sequential_graph(task_id)
    apply_runtime_checkpoint_to_task(
        task_a,
        RuntimeCheckpoint(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_tree=ExecutionTreeRecorder.start_root(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                root_execution_id=root_before,
            ).snapshot,
        ),
    )

    await _run_graph(
        executor=interrupt_executor,
        task=task_a,
        graph=graph,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root_before,
    )
    counts_after_interrupt = engine.snapshot_counts()
    assert counts_after_interrupt == _NodeInvocationCounts(agent_a=1, agent_b=1, agent_c=1, agent_d=0)

    runtime_before = resolve_task_runtime_checkpoint(task_a)
    assert runtime_before is not None
    execution_a_before = _execution_id_for_node(runtime_before, _NODE_A)
    execution_b_before = _execution_id_for_node(runtime_before, _NODE_B)
    execution_c_before = _execution_id_for_node(runtime_before, _NODE_C)
    assert execution_a_before is not None
    assert execution_b_before is not None
    assert execution_c_before is not None

    entry_a = runtime_before.execution_tree.entry_by_graph_node_id(_NODE_A)
    entry_b = runtime_before.execution_tree.entry_by_graph_node_id(_NODE_B)
    assert entry_a is not None and entry_a.status is ExecutionCheckpointStatus.COMPLETED
    assert entry_b is not None and entry_b.status is ExecutionCheckpointStatus.COMPLETED
    entry_c = runtime_before.execution_tree.entry_by_graph_node_id(_NODE_C)
    assert entry_c is not None
    assert entry_c.status is ExecutionCheckpointStatus.FAILED

    checkpoint = TaskCheckpoint(
        task_id=task_id,
        tenant_id="t1",
        resume_token="rt_ue_11e",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=build_runtime_checkpoint(
            task_a,
            run_id=run_id,
            attempt_id=attempt_id,
            graph=graph,
        ),
    )
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ue_11e_resume.db")
    store.save(checkpoint)
    loaded = store.get_by_token(task_id, "t1", "rt_ue_11e")
    assert loaded is not None and loaded.runtime is not None

    resumed_root = mint_execution_id()
    assert resumed_root != root_before

    task_b = Task(
        task_id=task_id,
        tenant_id="t1",
        user_id="u1",
        message="ue-11e resume continue",
        context=TaskContext(capability=_CAPABILITY),
    )
    prepare_task_for_checkpoint_resume(
        task_b,
        loaded,
        active_attempt_id=attempt_id,
        active_root_execution_id=resumed_root,
    )

    engine._gate_open = True
    resume_executor = GraphExecutor(
        registry,
        engine=engine,
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=0)),
    )
    resume_graph = _sequential_graph(task_id)

    await _run_graph(
        executor=resume_executor,
        task=task_b,
        graph=resume_graph,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=resumed_root,
    )

    counts_final = engine.snapshot_counts()
    assert counts_final.agent_a == 1
    assert counts_final.agent_b == 1
    assert counts_final.agent_c == 2
    assert counts_final.agent_d == 1

    runtime_after = resolve_task_runtime_checkpoint(task_b)
    assert runtime_after is not None
    assert runtime_after.run_id == run_id
    assert runtime_after.attempt_id == attempt_id

    entry_a_after = runtime_after.execution_tree.entry_by_graph_node_id(_NODE_A)
    entry_b_after = runtime_after.execution_tree.entry_by_graph_node_id(_NODE_B)
    assert entry_a_after is not None
    assert entry_a_after.execution_id == execution_a_before
    assert entry_b_after is not None
    assert entry_b_after.execution_id == execution_b_before

    root_entries = [
        entry
        for entry in runtime_after.execution_tree.entries
        if entry.parent_execution_id is None
    ]
    assert len(root_entries) == 1
    assert root_entries[0].execution_id == resumed_root

    execution_d = _execution_id_for_node(runtime_after, _NODE_D)
    assert execution_d is not None
    entry_d = runtime_after.execution_tree.entry_by_graph_node_id(_NODE_D)
    assert entry_d is not None
    assert entry_d.parent_execution_id == resumed_root
    assert entry_d.status is ExecutionCheckpointStatus.COMPLETED

    execution_c_after = _execution_id_for_node(runtime_after, _NODE_C)
    assert execution_c_after is not None
    assert execution_c_after != execution_c_before

    assert resume_graph.node_by_id(_NODE_D).status == ExecutionNodeStatus.COMPLETED
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None
    assert peek_active_execution_authority() is None
    assert peek_active_execution_budget() is None
