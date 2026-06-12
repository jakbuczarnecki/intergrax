# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.agent_handoff import AgentHandoff
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent


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
        UaepPipelineStubAgent(
            agent_id="agent_a",
            capability="cap.source",
            prefix="A",
            answer_separator=":",
            route_extra={"pending_handoff": handoff.model_dump(mode="json")},
            description="graph handoff/retry stub",
        )
    )
    registry.register(
        UaepPipelineStubAgent(
            agent_id="agent_b",
            capability="cap.handoff_target",
            prefix="B",
            answer_separator=":",
            description="graph handoff/retry stub",
        )
    )

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
    registry.register(
        UaepPipelineStubAgent(
            agent_id="agent_a",
            capability="cap.shared",
            prefix="A",
            answer_separator=":",
            description="graph handoff/retry stub",
        )
    )
    registry.register(
        UaepPipelineStubAgent(
            agent_id="agent_b",
            capability="cap.shared",
            prefix="B",
            answer_separator=":",
            description="graph handoff/retry stub",
        )
    )

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
