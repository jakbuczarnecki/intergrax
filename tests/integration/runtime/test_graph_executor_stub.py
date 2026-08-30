# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.graph_execution_context import bound_graph_execution_context
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.gate]


async def test_graph_executor_runs_two_agents_sequentially():
    registry = AgentRegistry()
    registry.register(UaepPipelineStubAgent(agent_id="agent_a", capability="cap.a", prefix="A"))
    registry.register(UaepPipelineStubAgent(agent_id="agent_b", capability="cap.b", prefix="B"))

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="pipeline",
        context=TaskContext(capability="cap.a"),
    )
    graph = ExecutionGraph(
        graph_id="graph_gate_1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.a"),
            ExecutionNode(
                node_id="n2",
                agent_id="agent_b",
                capability="cap.b",
                depends_on=["n1"],
            ),
        ],
    )

    executor = GraphExecutor(registry)
    with bound_graph_execution_context():
        executions, retries, graph, _ = await executor.execute(graph, task)

    assert retries == []
    assert len(executions) == 2
    assert executions[0].agent_id == "agent_a"
    assert executions[0].status == AgentExecutionStatus.COMPLETED
    assert executions[0].summary == "A: pipeline"
    assert executions[1].agent_id == "agent_b"
    assert executions[1].status == AgentExecutionStatus.COMPLETED
    assert executions[1].summary.startswith("B: pipeline")
    assert "A: pipeline" in executions[1].summary
    assert all(node.status == ExecutionNodeStatus.COMPLETED for node in graph.nodes)

    shared = task.metadata.get("shared_task_context")
    assert isinstance(shared, dict)
    assert shared["structured_outputs"]["n1"]["agent_id"] == "agent_a"
    assert shared["structured_outputs"]["n2"]["agent_id"] == "agent_b"
