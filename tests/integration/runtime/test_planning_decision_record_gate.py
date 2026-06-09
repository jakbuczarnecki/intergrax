# © Artur Czarnecki. All rights reserved.

import pytest

from echo.echo_agent import EchoAgent
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
    decision_events = [e for e in events if e.event_type is RuntimeEventType.DECISION_EMITTED]
    assert decision_events, "expected DECISION_EMITTED during planning"
    advisory = [
        e
        for e in events
        if e.event_type is RuntimeEventType.TASK_PROGRESS
        and e.payload.get("event_kind") == "COORDINATION_PATTERN_ADVISORY"
    ]
    assert advisory, "expected coordination advisory when emit_coordination_advisory=True"
