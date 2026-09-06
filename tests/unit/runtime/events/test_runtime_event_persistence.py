# © Artur Czarnecki. All rights reserved.

import pytest

pytestmark = pytest.mark.no_ci

from intergrax.contracts.execution_identity import mint_attempt_id, mint_event_id, mint_run_id, mint_task_id
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
from intergrax.runtime.events.stores.validating_runtime_event_store import (
    ValidatingRuntimeEventPersistence,
)
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore


def _sample_event(**updates: object) -> RuntimeEvent:
    from intergrax.contracts.execution_identity import mint_execution_id

    payload: dict[str, object] = {"human_request": {"urgency": "critical"}}
    fields: dict[str, object] = {
        "tenant_id": "t1",
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
        "event_type": RuntimeEventType.HUMAN_APPROVAL_REQUESTED,
        "phase": ExecutionPhase.HUMAN_APPROVAL,
        "payload": payload,
    }
    fields.update(updates)
    return RuntimeEvent.model_validate(fields)


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_runtime_event_persistence_disabled_by_default():
    assert resolve_runtime_event_persistence() is None


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_runtime_event_persistence_explicit_implementation():
    custom = InMemoryRuntimeEventStore()
    store = resolve_runtime_event_persistence(implementation=custom)
    assert isinstance(store, ValidatingRuntimeEventPersistence)
    assert store._inner is custom  # type: ignore[attr-defined]


@pytest.mark.unit
@pytest.mark.gate
def test_open_runtime_event_store_sqlite(tmp_path):
    store = open_runtime_event_store(tmp_path / "events.db")
    assert isinstance(store, SQLiteRuntimeEventStore)


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_runtime_event_persistence_with_explicit_path(tmp_path):
    store = resolve_runtime_event_persistence(db_path=tmp_path / "events.db")
    assert isinstance(store, ValidatingRuntimeEventPersistence)
    assert isinstance(store._inner, SQLiteRuntimeEventStore)  # type: ignore[attr-defined]


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
    loaded = store.list_for_run(event.run_id, tenant_id="t1")
    assert len(loaded) == 1
    assert loaded[0].event_id == event.event_id
    assert loaded[0].attempt_id == event.attempt_id
    assert loaded[0].payload["human_request"]["urgency"] == "critical"


@pytest.mark.unit
@pytest.mark.gate
def test_sqlite_runtime_event_store_roundtrip(tmp_path):
    store = open_runtime_event_store(tmp_path / "events.db")
    event = _sample_event()
    store.append(event, tenant_id="t1")
    by_task = store.list_for_task(event.task_id, tenant_id="t1")
    assert len(by_task) == 1
    assert by_task[0].tenant_id == "t1"
    assert by_task[0].attempt_id == event.attempt_id


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_event_bus_persists_on_publish_and_record():
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    event = _sample_event(event_type=RuntimeEventType.PAUSED)
    await bus.publish(event)
    assert len(bus.history) == 1
    assert len(store.list_for_run(event.run_id, tenant_id="t1")) == 1

    recorded = _sample_event(
        event_id=mint_event_id(),
        run_id=event.run_id,
        task_id=event.task_id,
        event_type=RuntimeEventType.RESUMED,
    )
    bus.record(recorded, tenant_id="t1")
    assert len(store.list_for_run(event.run_id, tenant_id="t1")) == 2


@pytest.mark.unit
@pytest.mark.gate
def test_null_runtime_event_persistence_get_by_event_id() -> None:
    store = NullRuntimeEventPersistence()
    event = _sample_event()
    positioned = store.append(event, tenant_id="t1")
    lookup = store.get_by_event_id(tenant_id="t1", event_id=event.event_id)
    assert lookup is not None
    assert lookup.position == positioned.position
    assert store.get_by_event_id(tenant_id="t2", event_id=event.event_id) is None
    assert store.get_by_event_id(tenant_id="t1", event_id=mint_event_id()) is None


def test_null_runtime_event_persistence_explicit_tenant_when_event_tenant_none() -> None:
    store = NullRuntimeEventPersistence()
    event = _sample_event(tenant_id=None)
    positioned = store.append(event, tenant_id="tenant-a")
    lookup = store.get_by_event_id(tenant_id="tenant-a", event_id=event.event_id)
    assert lookup is not None
    assert lookup.position == positioned.position
    assert store.get_by_event_id(tenant_id="tenant-b", event_id=event.event_id) is None
