# © Artur Czarnecki. All rights reserved.

from intergrax.utils import attribute_access
import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.uaep import UAEPExecutor
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep, TaskPlanner
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent


class _CancellingGraphExecutor(GraphExecutor):
    async def execute(self, graph, task, **kwargs):
        original_complete = kwargs.get("on_node_complete")

        def _on_complete(node):
            if node.node_id in {"n1", "step_1"}:
                CancellationCoordinator.request(task, reason="operator_cancel")
            if original_complete is not None:
                original_complete(node)

        kwargs["on_node_complete"] = _on_complete
        return await super().execute(graph, task, **kwargs)


class _TwoStepPlanner(TaskPlanner):
    def plan(self, task: Task, registry: AgentRegistry) -> NexusPlan:
        _ = registry
        return NexusPlan(
            task_id=task.task_id,
            classification=task.classification or "multi_agent",
            steps=[
                PlanStep(
                    step_id="step_1",
                    agent_id="agent_a",
                    capability="graph.cancel",
                    description="first step",
                ),
                PlanStep(
                    step_id="step_2",
                    agent_id="agent_b",
                    capability="graph.cancel",
                    description="second step",
                    depends_on=["step_1"],
                ),
            ],
            validation_criteria=["non_empty_summary"],
        )


class _MultiStepUaepAgent(Agent):
    step_runs = 0

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="uaep_cancel",
            name="UAEP cancel agent",
            description="two-step uaep agent",
            capabilities=["uaep.cancel"],
            max_steps=3,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = attribute_access.optional(task_context, "capability", None)
        if capability == "uaep.cancel":
            return CapabilityMatchResult(
                matched=True,
                agent_id="uaep_cancel",
                matched_capabilities=["uaep.cancel"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

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
        return [
            AgentStep(step_id="s1", step_name="s1", step_index=0),
            AgentStep(step_id="s2", step_name="s2", step_index=1),
        ]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _MultiStepUaepAgent.step_runs += 1
        return StepOutput(step_id=step.step_id, summary=f"done:{step.step_id}")


@pytest.mark.unit
@pytest.mark.gate
def test_cancellation_coordinator_request_and_clear():
    task = Task(tenant_id="t1", user_id="u1")
    assert not CancellationCoordinator.is_requested(task.metadata)
    CancellationCoordinator.request(task, reason="test")
    assert CancellationCoordinator.is_requested(task.metadata)
    assert task.metadata["cancellation_reason"] == "test"
    CancellationCoordinator.clear(task)
    assert not CancellationCoordinator.is_requested(task.metadata)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_graph_executor_cancels_before_second_node():
    UaepPipelineStubAgent.run_count = 0
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(agent_id="agent_a", capability="graph.cancel", prefix="A")
    )
    registry.register(
        UaepPipelineStubAgent(agent_id="agent_b", capability="graph.cancel", prefix="B")
    )

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="cancel mid graph",
        context=TaskContext(capability="graph.cancel"),
    )
    graph = ExecutionGraph(
        graph_id="graph_cancel_1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(node_id="n1", agent_id="agent_a", capability="graph.cancel"),
            ExecutionNode(
                node_id="n2",
                agent_id="agent_b",
                capability="graph.cancel",
                depends_on=["n1"],
            ),
        ],
    )

    executor = _CancellingGraphExecutor(registry)

    executions, _, graph, cancelled = await executor.execute(graph, task)

    assert cancelled is True
    assert len(executions) == 1
    assert UaepPipelineStubAgent.run_count == 1
    assert graph.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED
    assert graph.node_by_id("n2").status == ExecutionNodeStatus.SKIPPED
    assert graph.node_by_id("n2").metadata.get("cancelled") is True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_graph_cancellation():
    UaepPipelineStubAgent.run_count = 0
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(agent_id="agent_a", capability="graph.cancel", prefix="A")
    )
    registry.register(
        UaepPipelineStubAgent(agent_id="agent_b", capability="graph.cancel", prefix="B")
    )

    loop = NexusLoop(
        registry,
        planner=_TwoStepPlanner(),
        graph_executor=_CancellingGraphExecutor(registry),
    )
    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="cancel via nexus",
            context=TaskContext(capability="graph.cancel"),
        )
    )

    assert result.state == TaskState.CANCELLED
    assert UaepPipelineStubAgent.run_count == 1
    assert any(
        event.event_type == RuntimeEventType.CANCELLED for event in loop.event_bus.history
    )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_uaep_executor_stops_on_cancellation():
    _MultiStepUaepAgent.step_runs = 0
    agent = _MultiStepUaepAgent()
    executor = UAEPExecutor()
    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="sess_1",
        agent_id="uaep_cancel",
        message="cancel uaep",
        metadata={
            "task_id": "task_uaep_cancel",
            "run_id": "task_uaep_cancel",
            "cancellation_requested": True,
        },
    )

    answer, validation, _, _ = await executor.execute(agent, request)

    assert _MultiStepUaepAgent.step_runs == 0
    assert not validation.valid
    assert "task_cancelled" in validation.errors
    assert answer.answer == ""
