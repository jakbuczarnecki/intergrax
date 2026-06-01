# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.delegation import DelegationSpec
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task_memory.delegation_memory import TaskMemoryMetadataKey
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _PrefixPipeline(RuntimePipeline):
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    async def _inner_run(self, state):
        answer = f"{self._prefix}:{state.request.message}"
        state.raw_answer = answer
        state.runtime_answer = RuntimeAnswer(run_id=state.run_id, answer=answer)
        return state.runtime_answer


class _StubAgent(Agent):
    def __init__(self, agent_id: str, capability: str, prefix: str) -> None:
        self._agent_id = agent_id
        self._capability = capability
        self._prefix = prefix
        self.last_metadata: dict[str, object] = {}

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="delegation stub",
            capabilities=[self._capability],
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        if task_context.capability == self._capability:
            return CapabilityMatchResult(
                matched=True,
                agent_id=self._agent_id,
                matched_capabilities=[self._capability],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        self.last_metadata = dict(request.metadata)
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text=f"{self._prefix}:{request.message}"),
            enable_rag=False,
            production_mode=False,
        )
        config.pipeline = _PrefixPipeline(self._prefix)
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_graph_executor_delegation_sets_memory_and_parent_metadata() -> None:
    parent = _StubAgent("parent", "cap.parent", "P")
    child = _StubAgent("child", "cap.child", "C")
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
