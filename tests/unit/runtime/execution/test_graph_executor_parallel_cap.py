# © Artur Czarnecki. All rights reserved.

"""GraphExecutor batch concurrency cap (Phase ORCH-3)."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_max_parallel_nodes_limits_concurrent_batch_execution() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    active = 0
    peak = 0
    lock = asyncio.Lock()

    original_execute = GraphExecutor._execute_node

    async def _slow_execute_node(self, graph, task, node, prior_outputs, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.05)
        async with lock:
            active -= 1
        return await original_execute(self, graph, task, node, prior_outputs, **kwargs)

    GraphExecutor._execute_node = _slow_execute_node  # type: ignore[method-assign]
    try:
        executor = GraphExecutor(registry, max_parallel_nodes=1)
        graph = ExecutionGraph(
            graph_id="g1",
            task_id="task1",
            nodes=[
                ExecutionNode(node_id="n1", agent_id="echo", capability="echo.basic"),
                ExecutionNode(node_id="n2", agent_id="echo", capability="echo.basic"),
            ],
        )
        task = Task(
            tenant_id="t1",
            user_id="u1",
            message="parallel cap",
            context=TaskContext(capability="echo.basic"),
        )
        await executor.execute(graph, task)
        assert peak == 1
        assert all(node.status == ExecutionNodeStatus.COMPLETED for node in graph.nodes)
    finally:
        GraphExecutor._execute_node = original_execute  # type: ignore[method-assign]
