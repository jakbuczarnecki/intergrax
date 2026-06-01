# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.agent_handoff import AgentHandoff
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest, RouteInfo
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _PrefixPipeline(RuntimePipeline):
    def __init__(self, prefix: str, *, extra: dict[str, object] | None = None) -> None:
        self._prefix = prefix
        self._extra = extra or {}

    async def _inner_run(self, state):
        route = RouteInfo(extra=dict(self._extra))
        answer = f"{self._prefix}:{state.request.message}"
        state.raw_answer = answer
        state.runtime_answer = RuntimeAnswer(
            run_id=state.run_id,
            answer=answer,
            route=route,
        )
        return state.runtime_answer


class _GraphAgent(Agent):
    def __init__(self, agent_id: str, capability: str, prefix: str, *, route_extra: dict[str, object] | None = None) -> None:
        self._agent_id = agent_id
        self._capability = capability
        self._prefix = prefix
        self._route_extra = route_extra

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="graph handoff/retry stub",
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
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text=self._prefix),
            enable_rag=False,
            production_mode=False,
        )
        config.pipeline = _PrefixPipeline(self._prefix, extra=self._route_extra)
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


class _FailOnceValidation(NexusValidationEngine):
    def __init__(self, *, fail_agent: str) -> None:
        super().__init__()
        self._fail_agent = fail_agent
        self._failed: set[str] = set()

    def validate(self, execution, *, contract, capability=None, plan_criteria=None) -> ValidationResult:
        agent_id = contract.id
        if agent_id == self._fail_agent and agent_id not in self._failed:
            self._failed.add(agent_id)
            return ValidationResult(valid=False, errors=["simulated validation failure"])
        return super().validate(
            execution,
            contract=contract,
            capability=capability,
            plan_criteria=plan_criteria,
        )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_graph_executor_handoff_spawns_followup_node() -> None:
    handoff = AgentHandoff(
        from_agent_id="agent_a",
        to_capability="cap.handoff_target",
        reason="delegate downstream",
    )
    registry = AgentRegistry()
    registry.register(
        _GraphAgent(
            "agent_a",
            "cap.source",
            "A",
            route_extra={"pending_handoff": handoff.model_dump(mode="json")},
        )
    )
    registry.register(_GraphAgent("agent_b", "cap.handoff_target", "B"))

    task = Task(tenant_id="t1", user_id="u1", message="handoff", context=TaskContext(capability="cap.source"))
    graph = ExecutionGraph(
        graph_id="handoff_graph",
        task_id=task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.source")],
    )
    bus = RuntimeEventBus()
    executor = GraphExecutor(registry, event_bus=bus)
    executions, _, graph_out, _ = await executor.execute(graph, task)

    assert len(executions) >= 2
    assert executions[0].agent_id == "agent_a"
    assert executions[1].agent_id == "agent_b"
    assert graph_out.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED
    handoff_events = [
        e
        for e in bus.history
        if e.event_type in (RuntimeEventType.HANDOFF_INITIATED, RuntimeEventType.HANDOFF_COMPLETED)
    ]
    assert handoff_events


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_graph_executor_retries_with_alternate_agent() -> None:
    registry = AgentRegistry()
    registry.register(_GraphAgent("agent_a", "cap.shared", "A"))
    registry.register(_GraphAgent("agent_b", "cap.shared", "B"))

    task = Task(tenant_id="t1", user_id="u1", message="retry", context=TaskContext(capability="cap.shared"))
    graph = ExecutionGraph(
        graph_id="retry_graph",
        task_id=task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.shared")],
    )
    validation = _FailOnceValidation(fail_agent="agent_a")
    executor = GraphExecutor(
        registry,
        validation_engine=validation,
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=1)),
    )
    executions, retries, graph_out, _ = await executor.execute(graph, task)

    assert len(retries) == 1
    assert retries[0].alternate_agent_id == "agent_b"
    assert executions[-1].agent_id == "agent_b"
    assert graph_out.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED
