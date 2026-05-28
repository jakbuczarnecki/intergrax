# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.uaep import UAEPExecutor
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_handoff import AgentHandoff
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _HandoffSourceAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="handoff_source",
            name="Handoff Source",
            description="requests handoff",
            capabilities=["cap.source"],
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
        return [AgentStep(step_id="delegate", step_name="delegate", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = step, ctx
        return StepOutput(step_id=step.step_id, summary="source complete")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(
            type=AgentDecisionType.MODIFY_PLAN,
            reason="delegate to specialist",
            handoff=AgentHandoff(
                from_agent_id="handoff_source",
                to_capability="cap.target",
                reason="needs target agent",
                payload={"request": "analyze vendors"},
            ),
        )


class _HandoffTargetAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="handoff_target",
            name="Handoff Target",
            description="accepts handoff",
            capabilities=["cap.target"],
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
        return [AgentStep(step_id="accept", step_name="accept", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = step
        shared = ctx.metadata.get("shared_task_context") or {}
        handoffs = {
            key: value
            for key, value in (shared.get("structured_outputs") or {}).items()
            if str(key).startswith("handoff:")
        }
        return StepOutput(step_id=step.step_id, summary=f"target:{bool(handoffs)}")

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
async def test_graph_executor_runs_handoff_target_agent():
    bus = RuntimeEventBus()
    registry = AgentRegistry()
    registry.register(_HandoffSourceAgent())
    registry.register(_HandoffTargetAgent())

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="handoff flow",
        context=TaskContext(capability="cap.source"),
    )
    graph = ExecutionGraph(
        graph_id="graph_handoff_1",
        task_id=task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="handoff_source", capability="cap.source")],
    )

    engine = AgentEngine(registry, uaep_executor=UAEPExecutor(event_bus=bus))
    executor = GraphExecutor(
        registry,
        engine=engine,
        context_manager=ContextManager(),
        event_bus=bus,
    )

    executions, retries, graph, failed = await executor.execute(graph, task)

    assert failed is False
    assert retries == []
    assert len(executions) == 2
    assert executions[0].agent_id == "handoff_source"
    assert executions[1].agent_id == "handoff_target"
    assert executions[1].summary == "target:True"
    assert len(graph.nodes) == 2
    assert any(node.node_id.startswith("handoff_") for node in graph.nodes)
    assert any(e.event_type == RuntimeEventType.HANDOFF_INITIATED for e in bus.history)
    assert any(e.event_type == RuntimeEventType.HANDOFF_COMPLETED for e in bus.history)
