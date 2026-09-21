# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from testing_support.nexus_hitl_test_agent import NexusBasicHitlTestAgent
from testing_support.nexus_lab_task_execution import run_lab_nexus_task

_HitlAgent = NexusBasicHitlTestAgent


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_persists_runtime_events_via_injected_store():
    store = InMemoryRuntimeEventStore()
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    loop = NexusLoop(registry, runtime_event_store=store)

    paused = await run_lab_nexus_task(
        loop,
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
        ),
    )

    assert paused.state == TaskState.WAITING_FOR_HUMAN
    persisted = store.list_for_task(paused.task_id, tenant_id="t1")
    assert persisted
    assert all(event.tenant_id == "t1" for event in persisted)
    assert any(
        event.event_type == RuntimeEventType.HUMAN_APPROVAL_REQUESTED
        for event in persisted
    )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_persists_runtime_events_via_sqlite_path(tmp_path):
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    loop = NexusLoop(
        registry,
        runtime_events_db_path=tmp_path / "runtime_events.db",
    )

    paused = await run_lab_nexus_task(
        loop,
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
        ),
    )

    assert loop.runtime_event_store is not None
    persisted = loop.runtime_event_store.list_for_task(paused.task_id, tenant_id="t1")
    assert persisted
    assert any(
        event.event_type == RuntimeEventType.HUMAN_APPROVAL_REQUESTED
        for event in persisted
    )
