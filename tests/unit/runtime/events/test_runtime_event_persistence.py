# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import NullRuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.store import (
    open_runtime_event_store,
    resolve_runtime_event_persistence,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.events.stores.sqlite_runtime_event_store import (
    SQLiteRuntimeEventStore,
)


def _sample_event(**updates) -> RuntimeEvent:
    base = RuntimeEvent(
        tenant_id="t1",
        task_id="task_1",
        run_id="run_1",
        event_type=RuntimeEventType.HUMAN_APPROVAL_REQUESTED,
        phase=ExecutionPhase.HUMAN_APPROVAL,
        payload={"human_request": {"urgency": "critical"}},
    )
    return base.model_copy(update=updates)


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_runtime_event_persistence_disabled_by_default():
    assert resolve_runtime_event_persistence() is None


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_runtime_event_persistence_explicit_implementation():
    custom = InMemoryRuntimeEventStore()
    store = resolve_runtime_event_persistence(implementation=custom)
    assert store is custom


@pytest.mark.unit
@pytest.mark.gate
def test_open_runtime_event_store_sqlite(tmp_path):
    store = open_runtime_event_store(tmp_path / "events.db")
    assert isinstance(store, SQLiteRuntimeEventStore)


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_runtime_event_persistence_with_explicit_path(tmp_path):
    store = resolve_runtime_event_persistence(db_path=tmp_path / "events.db")
    assert isinstance(store, SQLiteRuntimeEventStore)


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_runtime_event_persistence_null_implementation():
    custom = NullRuntimeEventPersistence()
    store = resolve_runtime_event_persistence(implementation=custom)
    assert store is custom


@pytest.mark.unit
@pytest.mark.gate
def test_memory_runtime_event_store_roundtrip():
    store = InMemoryRuntimeEventStore()
    event = _sample_event()
    store.append(event, tenant_id="t1")
    loaded = store.list_for_run("run_1", tenant_id="t1")
    assert len(loaded) == 1
    assert loaded[0].event_id == event.event_id
    assert loaded[0].payload["human_request"]["urgency"] == "critical"


@pytest.mark.unit
@pytest.mark.gate
def test_sqlite_runtime_event_store_roundtrip(tmp_path):
    store = open_runtime_event_store(tmp_path / "events.db")
    event = _sample_event()
    store.append(event, tenant_id="t1")
    by_task = store.list_for_task("task_1", tenant_id="t1")
    assert len(by_task) == 1
    assert by_task[0].tenant_id == "t1"


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_event_bus_persists_on_publish_and_record():
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    event = _sample_event(event_type=RuntimeEventType.PAUSED)
    await bus.publish(event)
    assert len(bus.history) == 1
    assert len(store.list_for_run("run_1", tenant_id="t1")) == 1

    recorded = _sample_event(
        event_id="evt_record_only",
        event_type=RuntimeEventType.RESUMED,
    )
    bus.record(recorded, tenant_id="t1")
    assert len(store.list_for_run("run_1", tenant_id="t1")) == 2
