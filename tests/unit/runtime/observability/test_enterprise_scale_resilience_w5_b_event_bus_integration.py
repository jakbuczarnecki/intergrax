# © Artur Czarnecki. All rights reserved.

"""W5-B — RuntimeEventBus ↔ BoundedEventSink integration qualification."""

from __future__ import annotations

import threading
import time

import pytest

from intergrax.contracts.event_delivery import (
    CriticalEventDeliveryError,
    EventDeliveryDisposition,
    EventDeliveryObligation,
    EventDeliveryPolicy,
    EventDeliveryResult,
    EventPriority,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    InMemoryEventSink,
)
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _terminal_event() -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        **runtime_event_test_identity(),
    )


def _best_effort_event() -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
        **runtime_event_test_identity(),
    )


@pytest.mark.asyncio
async def test_runtime_event_bus_uses_sink() -> None:
    sink = InMemoryEventSink()
    bus = RuntimeEventBus(record_history=False, event_sink=sink)
    event = _terminal_event()
    await bus.publish(event)
    bus.close()
    assert len(sink.records) == 1
    assert sink.records[0][1].event_id == str(event.event_id)


def test_critical_event_cannot_disappear_when_buffer_full() -> None:
    gate = threading.Event()

    class _GatedDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            gate.wait(timeout=10.0)
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.COMPLETION,
            )

        def close(self) -> None:
            gate.set()

    bounded = BoundedEventSink(
        _GatedDownstream(),
        EventDeliveryPolicy(max_capacity=4),
    )
    bus = RuntimeEventBus(record_history=False, event_sink=bounded)
    errors: list[CriticalEventDeliveryError] = []
    lock = threading.Lock()
    start_barrier = threading.Barrier(7)

    def _record_critical() -> None:
        start_barrier.wait(timeout=5.0)
        try:
            bus.record(_terminal_event())
        except CriticalEventDeliveryError as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=_record_critical) for _ in range(6)]

    def _release_when_ready() -> None:
        start_barrier.wait(timeout=5.0)
        time.sleep(0.3)
        gate.set()

    threads.append(threading.Thread(target=_release_when_ready))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15.0)
    assert errors
    bus.close()


def test_best_effort_drop_when_buffer_full() -> None:
    downstream = InMemoryEventSink(consume_delay_seconds=1.0)
    bounded = BoundedEventSink(
        downstream,
        EventDeliveryPolicy(max_capacity=4),
    )
    bus = RuntimeEventBus(record_history=False, event_sink=bounded)
    for _ in range(20):
        bus.record(_best_effort_event())
    metrics = bus.delivery_metrics
    assert metrics is not None
    snap = metrics.snapshot()
    assert snap.events_dropped > 0
    bus.close()


def test_execution_isolation_when_consumer_blocked() -> None:
    downstream = InMemoryEventSink(consume_delay_seconds=0.2)
    bounded = BoundedEventSink(
        downstream,
        EventDeliveryPolicy(max_capacity=8),
    )
    bus = RuntimeEventBus(record_history=False, event_sink=bounded)
    start = time.monotonic()
    for _ in range(80):
        bus.record(_best_effort_event())
    elapsed = time.monotonic() - start
    bus.close()
    assert elapsed < 2.0


def test_shutdown_safety_bus_close_stops_consumer() -> None:
    downstream = InMemoryEventSink(consume_delay_seconds=0.05)
    bounded = BoundedEventSink(
        downstream,
        EventDeliveryPolicy(max_capacity=8),
    )
    bus = RuntimeEventBus(record_history=False, event_sink=bounded)
    errors: list[BaseException] = []

    def _publisher() -> None:
        try:
            for _ in range(30):
                bus.record(_terminal_event())
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=_publisher)
    worker.start()
    time.sleep(0.05)
    bus.close()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert downstream.records


def test_metrics_isolation_no_recursive_event_creation() -> None:
    sink = InMemoryEventSink()
    bus = RuntimeEventBus(record_history=False, event_sink=sink)
    bus.record(_best_effort_event())
    bus.record(_best_effort_event())
    metrics = bus.delivery_metrics
    assert metrics is not None
    snap = metrics.snapshot()
    assert snap.events_accepted == 2
    assert len(sink.records) == 2
    bus.close()
