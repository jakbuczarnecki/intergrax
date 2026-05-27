# © Artur Czarnecki. All rights reserved.

import pytest

from echo.echo_agent import EchoAgent
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_runs_echo_agent():
    registry = AgentRegistry()
    agent = EchoAgent()
    registry.register(agent)

    loop = NexusLoop(registry)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hello harness",
        context=TaskContext(capability="echo.basic"),
    )

    result = await loop.handle_task(task)

    assert result.state == TaskState.COMPLETED
    assert "hello harness" in result.answer
    assert result.agent_id == "echo"
    assert result.metadata.get("validation_valid") is True
    assert result.metadata.get("runtime_events", 0) > 0
    assert len(loop.event_bus.history) > 0
    assert loop.event_bus.history[0].event_type == RuntimeEventType.TASK_CREATED
    assert loop.event_bus.history[-1].event_type == RuntimeEventType.TASK_COMPLETED


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_echo_emits_lifecycle_trace():
    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(registry)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="lifecycle check",
        context=TaskContext(capability="echo.basic"),
    )

    await loop.handle_task(task)

    emitter = loop.trace_emitter
    assert emitter is not None
    lifecycle_states = {
        event.tags.get("task_state")
        for event in emitter.events
        if event.step == "task_lifecycle"
    }
    assert TaskState.CLASSIFIED.value in lifecycle_states
    assert TaskState.PLANNED.value in lifecycle_states
    assert TaskState.RUNNING.value in lifecycle_states
    assert TaskState.VALIDATING.value in lifecycle_states
    assert TaskState.COMPLETED.value in lifecycle_states
