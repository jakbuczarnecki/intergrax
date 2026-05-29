# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.debug.store import open_trace_reader
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState

pytestmark = [pytest.mark.integration, pytest.mark.gate]


@pytest.mark.asyncio
async def test_wire_nexus_observability_sqlite_trace_persists(tmp_path) -> None:
    trace_db = tmp_path / "trace.db"
    events_db = tmp_path / "events.db"
    stores = wire_nexus_observability(
        trace_db_path=trace_db,
        runtime_events_db_path=events_db,
    )

    assert isinstance(stores.trace_store, SQLiteRunTraceStore)
    assert stores.trace_db_path == trace_db
    assert stores.runtime_events_db_path == events_db
    assert stores.runtime_event_store is not None

    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(
        registry,
        trace_store=stores.trace_store,
        runtime_event_store=stores.runtime_event_store,
    )
    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="observability wiring",
            context=TaskContext(capability="echo.basic"),
        )
    )
    assert result.state == TaskState.COMPLETED

    persisted = open_trace_reader(trace_db).read_run(result.task_id, "t1")
    assert persisted.metadata.run_id == result.task_id
    assert len(persisted.events) > 0

    events = stores.runtime_event_store.list_for_task(result.task_id, tenant_id="t1")
    assert len(events) > 0


def test_wire_nexus_observability_in_memory_opt_out() -> None:
    stores = wire_nexus_observability(use_in_memory_trace=True, enable_runtime_events=False)
    assert isinstance(stores.trace_store, InMemoryRunTraceStore)
    assert stores.trace_db_path is None
    assert stores.runtime_event_store is None
