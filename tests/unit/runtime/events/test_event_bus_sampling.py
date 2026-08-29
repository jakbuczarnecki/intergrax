# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import get_catalog_entry, should_persist_event
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore

pytestmark = pytest.mark.gate


def _progress_event(event_id: str) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=event_id,
        event_type=RuntimeEventType.TASK_PROGRESS,
        phase=ExecutionPhase.STEP_EXECUTION,
        **runtime_event_test_identity(),
    )


def test_should_persist_event_respects_catalog_sample_rate() -> None:
    entry = get_catalog_entry(RuntimeEventType.TASK_PROGRESS)
    assert entry is not None
    assert entry.sample_rate < 1.0
    accepted = [i for i in range(500) if should_persist_event(_progress_event(f"evt_{i}"))]
    ratio = len(accepted) / 500
    assert 0.10 <= ratio <= 0.40


def test_bus_always_records_history_but_samples_persistence() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store, record_history=True)
    for i in range(200):
        bus.record(_progress_event(f"evt_sample_{i}"))
    assert len(bus.history) == 200
    persisted = store.list_for_task("task-1", tenant_id="")
    assert len(persisted) < len(bus.history)
    assert len(persisted) > 0


def test_bus_persists_full_rate_spine_events() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store, record_history=True)
    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
    )
    bus.record(event)
    assert len(store.list_for_task("task-1", tenant_id="")) == 1
