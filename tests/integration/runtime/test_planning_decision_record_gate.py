# © Artur Czarnecki. All rights reserved.

import pytest

from echo.echo_agent import EchoAgent
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState

pytestmark = [pytest.mark.integration, pytest.mark.gate]


@pytest.mark.asyncio
async def test_planning_emits_decision_record_event() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(
        registry,
        emit_coordination_advisory=True,
    )

    task = Task(
        tenant_id="t",
        user_id="u",
        message="hello",
        context=TaskContext(capability="echo.basic"),
    )
    result = await loop.handle_task(task)
    assert result.state == TaskState.COMPLETED
    events = loop._event_bus.history  # noqa: SLF001 — gate inspects published spine
    plan_events = [e for e in events if e.event_type is RuntimeEventType.PLAN_CREATED]
    assert plan_events, "expected PLAN_CREATED during planning"
    planning_event = next(
        (event for event in plan_events if "decision_record" in event.payload),
        None,
    )
    assert planning_event is not None, "expected decision record on explicit plan event"
    planning_decision = planning_event.payload["decision_record"]
    assert planning_decision.get("decision_type") == "nexus_planning"
    assert planning_decision.get("policy_action") == "allow"
    assert "classification" in planning_decision.get("metadata", {})
    assert "planner_source" in planning_decision.get("metadata", {})
    decision_events = [
        event
        for event in events
        if event.event_type is RuntimeEventType.DECISION_EMITTED
    ]
    assert not any(
        event.phase is ExecutionPhase.PLANNING for event in decision_events
    ), "planning must not emit step-level DECISION_EMITTED"
    advisory = [
        e
        for e in events
        if e.event_type is RuntimeEventType.TASK_PROGRESS
        and e.payload.get("event_kind") == "COORDINATION_PATTERN_ADVISORY"
    ]
    assert advisory, "expected coordination advisory when emit_coordination_advisory=True"
