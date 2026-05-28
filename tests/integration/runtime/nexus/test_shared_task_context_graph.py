# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.context.shared_task_context import load_shared_task_context_from_metadata
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task_memory import InMemoryTaskMemoryStore
from intergrax.agents.uaep import UAEPExecutor
from intergrax.agents.agent_engine import AgentEngine
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _ProducerAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="producer",
            name="Producer",
            description="writes shared output",
            capabilities=["cap.produce"],
            max_steps=1,
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="ok"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [AgentStep(step_id="produce", step_name="produce", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = step
        return StepOutput(step_id=step.step_id, summary="producer summary")


class _ConsumerAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="consumer",
            name="Consumer",
            description="reads shared output",
            capabilities=["cap.consume"],
            max_steps=1,
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="ok"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [AgentStep(step_id="consume", step_name="consume", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = step
        shared = load_shared_task_context_from_metadata(ctx.metadata)
        assert shared is not None
        assert "n1" in shared.structured_outputs
        assert shared.structured_outputs["n1"]["summary"] == "producer summary"

        memory_summary = None
        if ctx.memory_view is not None:
            payload = await ctx.memory_view.read("shared", "n1")
            memory_summary = payload.get("summary") if payload else None

        return StepOutput(
            step_id=step.step_id,
            summary=f"consumer:{memory_summary or shared.structured_outputs['n1']['summary']}",
        )

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_graph_executor_populates_shared_task_context():
    registry = AgentRegistry()
    registry.register(_ProducerAgent())
    registry.register(_ConsumerAgent())

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="cross-agent",
        context=TaskContext(capability="cap.produce"),
    )
    graph = ExecutionGraph(
        graph_id="graph_shared_1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(node_id="n1", agent_id="producer", capability="cap.produce"),
            ExecutionNode(
                node_id="n2",
                agent_id="consumer",
                capability="cap.consume",
                depends_on=["n1"],
            ),
        ],
    )

    memory = InMemoryTaskMemoryStore()
    engine = AgentEngine(
        registry,
        uaep_executor=UAEPExecutor(event_bus=RuntimeEventBus(), task_memory_store=memory),
    )
    executor = GraphExecutor(registry, engine=engine, context_manager=ContextManager())
    executions, retries, _, _ = await executor.execute(graph, task)

    assert retries == []
    assert len(executions) == 2
    assert executions[1].summary == "consumer:producer summary"

    shared = load_shared_task_context_from_metadata(task.metadata)
    assert shared is not None
    assert shared.structured_outputs["n1"]["summary"] == "producer summary"
    assert shared.structured_outputs["n2"]["agent_id"] == "consumer"
    assert shared.version >= 2

    persisted = memory.get(
        tenant_id="t1",
        task_id=task.task_id,
        namespace="shared",
        key="n1",
    )
    assert persisted is not None
    assert persisted.value["summary"] == "producer summary"
