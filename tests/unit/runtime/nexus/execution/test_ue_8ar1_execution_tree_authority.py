# © Artur Czarnecki. All rights reserved.

"""UE-8AR1 — execution-tree authority ownership (no graph topology bypass)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import AsyncMock, patch

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.delegation import DelegationSpec
from intergrax.contracts.delegation_authority import (
    EFFECTIVE_PERMISSION_SCOPES_METADATA_KEY,
    ParentExecutionAuthority,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    peek_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@contextmanager
def _bound_orchestration(
    *,
    scopes: tuple[str, ...] = ("read", "write"),
) -> Iterator[None]:
    execution_id = mint_execution_id()
    identity_token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.scoped(scopes),
    )
    budget_token = bind_root_execution_budget(
        execution_id=execution_id,
        ledger=create_execution_budget_ledger(None),
    )
    try:
        yield
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)


@pytest.mark.asyncio
async def test_graph_executor_without_active_execution_id_fails_closed() -> None:
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(
            agent_id="child",
            capability="cap.child",
            prefix="C",
        )
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="child")],
    )
    executor = GraphExecutor(registry)
    identity_token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    delegate_spy = AsyncMock(side_effect=AssertionError("delegate must not run"))
    try:
        with patch.object(
            ChildExecutionRunner,
            "execute",
            delegate_spy,
        ):
            with pytest.raises(RuntimeError, match="active ExecutionId required"):
                await executor.execute(graph, task)
    finally:
        reset_active_execution_identity(identity_token)
    delegate_spy.assert_not_called()


@pytest.mark.asyncio
async def test_graph_topology_depends_on_does_not_narrow_sibling_authority() -> None:
    node_a = UaepPipelineStubAgent(
        agent_id="agent_a",
        capability="cap.a",
        prefix="A",
        track_request_metadata=True,
    )
    node_b = UaepPipelineStubAgent(
        agent_id="agent_b",
        capability="cap.b",
        prefix="B",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(node_a)
    registry.register(node_b)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.a"),
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    graph = ExecutionGraph(
        graph_id="topology",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="A",
                agent_id="agent_a",
                capability="cap.a",
                delegation=DelegationSpec(
                    child_agent_id="agent_a",
                    permission_scopes=("read",),
                ),
            ),
            ExecutionNode(
                node_id="B",
                agent_id="agent_b",
                capability="cap.b",
                depends_on=["A"],
                delegation=DelegationSpec(
                    child_agent_id="agent_b",
                    permission_scopes=("write",),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry)
    with _bound_orchestration(scopes=("read", "write")):
        await executor.execute(graph, task)

    assert node_a.last_request is not None
    assert node_a.last_request.effective_delegation_authority is not None
    assert node_a.last_request.effective_delegation_authority.effective_permission_scopes == (
        "read",
    )
    assert node_b.last_request is not None
    assert node_b.last_request.effective_delegation_authority is not None
    assert node_b.last_request.effective_delegation_authority.effective_permission_scopes == (
        "write",
    )


@pytest.mark.asyncio
async def test_parallel_children_authority_isolated() -> None:
    left = UaepPipelineStubAgent(
        agent_id="left",
        capability="cap.left",
        prefix="L",
        track_request_metadata=True,
    )
    right = UaepPipelineStubAgent(
        agent_id="right",
        capability="cap.right",
        prefix="R",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(left)
    registry.register(right)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.left"),
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    graph = ExecutionGraph(
        graph_id="parallel",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="left",
                agent_id="left",
                capability="cap.left",
                delegation=DelegationSpec(
                    child_agent_id="left",
                    permission_scopes=("read",),
                ),
            ),
            ExecutionNode(
                node_id="right",
                agent_id="right",
                capability="cap.right",
                delegation=DelegationSpec(
                    child_agent_id="right",
                    permission_scopes=("write",),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry, max_parallel_nodes=2)
    with _bound_orchestration(scopes=("read", "write")):
        await executor.execute(graph, task)

    assert left.last_request is not None
    assert left.last_request.effective_delegation_authority is not None
    assert left.last_request.effective_delegation_authority.effective_permission_scopes == (
        "read",
    )
    assert right.last_request is not None
    assert right.last_request.effective_delegation_authority is not None
    assert right.last_request.effective_delegation_authority.effective_permission_scopes == (
        "write",
    )


@pytest.mark.asyncio
async def test_child_without_delegation_inherits_parent_authority() -> None:
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(child)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    graph = ExecutionGraph(
        graph_id="inherit",
        task_id=task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="child")],
    )
    executor = GraphExecutor(registry)
    with _bound_orchestration(scopes=("read", "write")):
        await executor.execute(graph, task)

    assert child.last_metadata.get(EFFECTIVE_PERMISSION_SCOPES_METADATA_KEY) is None
    assert peek_active_execution_authority() is None


@pytest.mark.asyncio
async def test_parent_authority_restored_after_child() -> None:
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
    )
    registry = AgentRegistry()
    registry.register(child)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    graph = ExecutionGraph(
        graph_id="restore",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("read",),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry)
    with _bound_orchestration(scopes=("read", "write")):
        await executor.execute(graph, task)

    assert peek_active_execution_authority() is None
