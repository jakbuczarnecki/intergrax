# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.delegation import DelegationSpec
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task_memory.delegation_memory import TaskMemoryMetadataKey
from testing_support.graph_execution_context import bound_graph_execution_context
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.gate]


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_graph_executor_delegation_sets_memory_and_parent_metadata() -> None:
    parent = UaepPipelineStubAgent(
        agent_id="parent",
        capability="cap.parent",
        prefix="P",
        answer_separator=":",
        track_request_metadata=True,
        description="delegation stub",
    )
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
        answer_separator=":",
        track_request_metadata=True,
        description="delegation stub",
    )
    registry = AgentRegistry()
    registry.register(parent)
    registry.register(child)

    task = Task(tenant_id="t1", user_id="u1", message="go", context=TaskContext(capability="cap.parent"))
    graph = ExecutionGraph(
        graph_id="delegation_graph",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(node_id="n1", agent_id="parent", capability="cap.parent"),
            ExecutionNode(
                node_id="n2",
                agent_id="child",
                capability="cap.child",
                depends_on=["n1"],
                delegation=DelegationSpec(
                    child_agent_id="child",
                    parent_run_id="parent-run",
                    parent_node_id="n1",
                ),
            ),
        ],
    )

    bus = RuntimeEventBus(record_history=True)
    executor = GraphExecutor(registry, event_bus=bus)
    with bound_graph_execution_context():
        executions, retries, completed_graph, _ = await executor.execute(graph, task)

    assert retries == []
    assert len(executions) == 2
    assert executions[1].status == AgentExecutionStatus.COMPLETED
    assert child.last_metadata[TaskMemoryMetadataKey.DELEGATION_MEMORY_NAMESPACE] == (
        f"{task.task_id}/delegation/n2"
    )
    assert child.last_metadata[TaskMemoryMetadataKey.PARENT_RUN_ID] == "parent-run"
    assert child.last_metadata[TaskMemoryMetadataKey.PARENT_NODE_ID] == "n1"
    assert all(node.status == ExecutionNodeStatus.COMPLETED for node in completed_graph.nodes)
    assert any(event.event_type == RuntimeEventType.CONTEXT_ASSEMBLED for event in bus.history)
