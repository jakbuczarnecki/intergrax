# © Artur Czarnecki. All rights reserved.

"""UE-10R4 — GraphExecutor canonical authority fail-closed behavior."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import AsyncMock, patch

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.delegation_authority import (
    EXECUTION_PERMISSION_SCOPES_METADATA_KEY,
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
    reset_active_execution_authority,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.graph_execution_context import bound_graph_execution_context
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@contextmanager
def _bound_identity_and_budget_without_authority() -> Iterator[None]:
    execution_id = mint_execution_id()
    identity_token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    budget_token = bind_root_execution_budget(
        execution_id=execution_id,
        ledger=create_execution_budget_ledger(None),
    )
    try:
        yield
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)


@pytest.mark.asyncio
async def test_graph_executor_without_active_authority_fails_closed() -> None:
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(
            agent_id="child",
            capability="cap.child",
            prefix="C",
        ),
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
    delegate_spy = AsyncMock(side_effect=AssertionError("delegate must not run"))
    with _bound_identity_and_budget_without_authority():
        with patch.object(ChildExecutionRunner, "execute", delegate_spy):
            with pytest.raises(RuntimeError, match="active execution authority required"):
                await executor.execute(graph, task)
    delegate_spy.assert_not_called()


@pytest.mark.asyncio
async def test_graph_executor_with_canonical_authority_executes_child() -> None:
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(child)
    authority = ParentExecutionAuthority.scoped(("read", "write"))
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        metadata={EXECUTION_PERMISSION_SCOPES_METADATA_KEY: ["read", "write"]},
        execution_authority=authority,
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="child")],
    )
    executor = GraphExecutor(registry)
    with bound_graph_execution_context(authority=authority):
        executions, _, graph_out, _ = await executor.execute(graph, task)
    assert executions[0].status == AgentExecutionStatus.COMPLETED
    assert graph_out.node_by_id("n1").status.value == "completed"
    assert child.last_request is not None


@pytest.mark.asyncio
async def test_graph_executor_rejects_task_authority_metadata_mismatch() -> None:
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(
            agent_id="child",
            capability="cap.child",
            prefix="C",
        ),
    )
    canonical_authority = ParentExecutionAuthority.scoped(("read",))
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        metadata={EXECUTION_PERMISSION_SCOPES_METADATA_KEY: ["read", "write"]},
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="child")],
    )
    executor = GraphExecutor(registry)
    delegate_spy = AsyncMock(side_effect=AssertionError("delegate must not run"))
    with bound_graph_execution_context(authority=canonical_authority):
        with patch.object(ChildExecutionRunner, "execute", delegate_spy):
            executions, _, _, _ = await executor.execute(graph, task)
    delegate_spy.assert_not_called()
    assert executions[0].status == AgentExecutionStatus.FAILED
    assert executions[0].errors
    assert "conflict" in executions[0].errors[0]


@pytest.mark.asyncio
async def test_graph_executor_retry_callback_is_awaited() -> None:
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(
            agent_id="child",
            capability="cap.child",
            prefix="C",
        ),
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="child")],
    )
    observed: list[str] = []

    async def on_retry(record) -> None:
        observed.append(record.reason)

    executor = GraphExecutor(registry)
    with bound_graph_execution_context():
        await executor.execute(graph, task, on_retry=on_retry)
    assert observed == []
