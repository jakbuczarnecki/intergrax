# © Artur Czarnecki. All rights reserved.

"""OBS-RUNTIME-HISTORY-BOUNDS: process-local RuntimeEventBus history tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event_history import (
    DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY,
    RuntimeEventHistoryBuffer,
    RuntimeEventHistoryPolicy,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.runtime_event_history import BoundedRuntimeEventHistory
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.observability.event_delivery.in_memory_sink import InMemoryEventSink
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_EVENT_BUS_PATH = _REPO / "intergrax" / "runtime" / "events" / "event_bus.py"


def _event(*, label: str, run_id: str | None = None) -> RuntimeEvent:
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
    )

    rid = run_id or mint_run_id()
    return RuntimeEvent.model_validate(
        {
            "tenant_id": "tenant-a",
            "task_id": mint_task_id(),
            "run_id": rid,
            "attempt_id": mint_attempt_id(),
            "execution_id": mint_execution_id(),
            "event_id": mint_event_id(),
            "event_type": RuntimeEventType.STEP_STARTED,
            "phase": ExecutionPhase.STEP_EXECUTION,
            "payload": {"label": label},
        },
    )


class CustomRuntimeEventHistory:
    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list[RuntimeEvent] = []

    def append(self, event: RuntimeEvent) -> None:
        if len(self._items) >= 2:
            self._items.pop(0)
        self._items.append(event)

    def snapshot(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._items)

    def clear(self) -> None:
        self._items.clear()


def test_default_bus_is_bounded() -> None:
    bus = RuntimeEventBus()
    for index in range(DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY + 50):
        bus.record(_event(label=str(index)))
    assert len(bus.history) == DEFAULT_BOUNDED_RUNTIME_EVENT_HISTORY_CAPACITY


def test_disabled_history() -> None:
    store = InMemoryRuntimeEventStore()
    sink = InMemoryEventSink()
    seen: list[RuntimeEvent] = []
    bus = RuntimeEventBus(
        persistence=store,
        record_history=False,
        event_sink=sink,
    )
    bus.subscribe(lambda event: seen.append(event))

    event = _event(label="only")
    bus.record(event, tenant_id="tenant-a")
    assert bus.history == []
    assert len(store.list_for_run(event.run_id, tenant_id="tenant-a")) == 1
    assert len(sink.records) == 1
    assert len(seen) == 1


def test_bounded_overflow_keeps_latest() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(3))
    events = [_event(label=str(i)) for i in range(5)]
    for item in events:
        bus.record(item)
    snapshot = bus.history
    assert [e.payload["label"] for e in snapshot] == ["2", "3", "4"]


def test_large_volume_stays_bounded() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(16))
    for index in range(10_000):
        bus.record(_event(label=str(index)))
    assert len(bus.history) == 16


@pytest.mark.asyncio
async def test_publish_respects_bound() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(2))
    await bus.publish(_event(label="a"))
    await bus.publish(_event(label="b"))
    await bus.publish(_event(label="c"))
    assert [e.payload["label"] for e in bus.history] == ["b", "c"]


def test_mixed_record_and_publish_respects_bound() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(2))
    bus.record(_event(label="r1"))
    asyncio.run(bus.publish(_event(label="p1")))
    bus.record(_event(label="r2"))
    assert [e.payload["label"] for e in bus.history] == ["p1", "r2"]


def test_clear_history() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(4))
    bus.record(_event(label="x"))
    bus.clear_history()
    assert bus.history == []
    bus.record(_event(label="y"))
    assert len(bus.history) == 1


def test_history_snapshot_is_copy() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(4))
    bus.record(_event(label="one"))
    first = bus.history
    second = bus.history
    assert first is not second
    assert first[0].event_id == second[0].event_id


def test_persistence_independent_of_local_eviction() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    task_id = mint_task_id()
    bus = RuntimeEventBus(
        persistence=store,
        history_policy=RuntimeEventHistoryPolicy.bounded(1),
    )
    for label in ("E1", "E2", "E3"):
        event = _event(label=label, run_id=run_id).model_copy(
            update={"task_id": task_id}
        )
        bus.record(event, tenant_id="tenant-a")
    assert [e.payload["label"] for e in bus.history] == ["E3"]
    persisted = store.list_for_run(run_id, tenant_id="tenant-a")
    assert [e.payload["label"] for e in persisted] == ["E1", "E2", "E3"]


def test_reconstruction_unaffected_by_local_eviction() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    task_id = mint_task_id()
    bus = RuntimeEventBus(
        persistence=store,
        history_policy=RuntimeEventHistoryPolicy.bounded(1),
    )
    for label in ("E1", "E2", "E3"):
        event = _event(label=label, run_id=run_id)
        event = event.model_copy(update={"task_id": task_id})
        bus.record(event, tenant_id="tenant-a")
    recon = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    view = recon.reconstruct_execution("tenant-a", task_id, run_id)
    assert len(view.positioned_events) == 3


def test_sink_and_handlers_unaffected_by_capacity() -> None:
    store = InMemoryRuntimeEventStore()
    sink = InMemoryEventSink()
    handled: list[str] = []
    bus = RuntimeEventBus(
        persistence=store,
        history_policy=RuntimeEventHistoryPolicy.bounded(1),
        event_sink=sink,
    )
    bus.subscribe(lambda event: handled.append(str(event.payload["label"])))
    for label in ("a", "b", "c"):
        bus.record(_event(label=label), tenant_id="tenant-a")
    assert len(sink.records) == 3
    assert handled == ["a", "b", "c"]
    assert len(bus.history) == 1


def test_conflicting_record_history_and_policy() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        RuntimeEventBus(
            record_history=True, history_policy=RuntimeEventHistoryPolicy.disabled()
        )


def test_conflicting_buffer_and_policy() -> None:
    with pytest.raises(ValueError, match="history_buffer cannot"):
        RuntimeEventBus(
            history_buffer=BoundedRuntimeEventHistory(4),
            history_policy=RuntimeEventHistoryPolicy.bounded(4),
        )


def test_invalid_bounded_policy_capacity() -> None:
    with pytest.raises(ValueError, match="max_events must be > 0"):
        RuntimeEventHistoryPolicy.bounded(0)


def test_custom_history_buffer_injection() -> None:
    custom: RuntimeEventHistoryBuffer = CustomRuntimeEventHistory()
    bus = RuntimeEventBus(history_buffer=custom)
    for label in ("1", "2", "3"):
        bus.record(_event(label=label))
    assert [e.payload["label"] for e in bus.history] == ["2", "3"]


def test_event_bus_has_no_unbounded_list_storage() -> None:
    source = _EVENT_BUS_PATH.read_text(encoding="utf-8")
    assert "_history: List" not in source
    assert "_history.append" not in source
    assert "_record_history" not in source


def test_history_implementations_are_not_persistence_ports() -> None:
    from intergrax.contracts.execution_evidence.persistence_port import (
        EvidencePersistencePort,
    )

    assert not isinstance(BoundedRuntimeEventHistory(4), EvidencePersistencePort)
    from intergrax.runtime.events.runtime_event_history import (
        DisabledRuntimeEventHistory,
    )

    assert not isinstance(DisabledRuntimeEventHistory(), EvidencePersistencePort)
