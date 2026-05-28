# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _AnswerPipeline(RuntimePipeline):
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        answer = f"{self._prefix}: {state.request.message}"
        state.raw_answer = answer
        state.runtime_answer = RuntimeAnswer(run_id=state.run_id, answer=answer)
        return state.runtime_answer


class _SequentialStubAgent(Agent):
    def __init__(self, *, agent_id: str, capability: str, prefix: str) -> None:
        self._agent_id = agent_id
        self._capability = capability
        self._prefix = prefix

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="graph gate stub",
            capabilities=[self._capability],
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability == self._capability:
            return CapabilityMatchResult(
                matched=True,
                agent_id=self._agent_id,
                matched_capabilities=[self._capability],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
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


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_graph_executor_runs_two_agents_sequentially():
    registry = AgentRegistry()
    registry.register(_SequentialStubAgent(agent_id="agent_a", capability="cap.a", prefix="A"))
    registry.register(_SequentialStubAgent(agent_id="agent_b", capability="cap.b", prefix="B"))

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
