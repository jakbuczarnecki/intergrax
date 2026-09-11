# © Artur Czarnecki. All rights reserved.

"""W5-A — observability event delivery backpressure qualification."""

from __future__ import annotations

import threading
import time

import pytest

from intergrax.contracts.event_delivery import (
    CriticalEventDeliveryError,
    CriticalEventKind,
    DeliverableEvent,
    EventDeliveryDisposition,
    EventDeliveryPolicy,
    EventPriority,
)
from intergrax.runtime.observability.event_delivery import BoundedEventSink, InMemoryEventSink

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _event(label: str, seq: int = 0) -> DeliverableEvent:
    return DeliverableEvent(event_id=label, kind=label, sequence=seq)


def test_best_effort_overflow_allows_drop_execution_continues() -> None:
    downstream = InMemoryEventSink(consume_delay_seconds=0.5)
    policy = EventDeliveryPolicy(max_capacity=10)
    sink = BoundedEventSink(downstream, policy)
    accepted = 0
    dropped = 0
    start = time.monotonic()
    for i in range(1000):
        result = sink.publish(
            _event(f"be-{i}"),
            priority=EventPriority.BEST_EFFORT,
        )
        if result.disposition is EventDeliveryDisposition.DROPPED:
            dropped += 1
        elif result.disposition is EventDeliveryDisposition.ACCEPTED:
            accepted += 1
    elapsed = time.monotonic() - start
    sink.close()
    assert dropped > 0
    assert accepted <= policy.max_capacity + 50
    assert elapsed < 2.0


def test_critical_overflow_fail_closed_no_silent_loss() -> None:
    downstream = InMemoryEventSink(consume_delay_seconds=1.0)
    policy = EventDeliveryPolicy(max_capacity=10)
    sink = BoundedEventSink(downstream, policy)
    for i in range(10):
        sink.publish(
            _event(f"c-{i}", seq=i),
            priority=EventPriority.CRITICAL,
        )
    with pytest.raises(CriticalEventDeliveryError):
        sink.publish(
            _event("c-overflow"),
            priority=EventPriority.CRITICAL,
        )
    sink.close()


def test_slow_consumer_does_not_block_producer() -> None:
    downstream = InMemoryEventSink(consume_delay_seconds=0.2)
    sink = BoundedEventSink(
        downstream,
        EventDeliveryPolicy(max_capacity=10),
    )
    start = time.monotonic()
    for i in range(100):
        sink.publish(_event(f"iso-{i}"), priority=EventPriority.BEST_EFFORT)
    elapsed = time.monotonic() - start
    sink.close()
    assert elapsed < 1.5


def test_critical_events_preserve_order() -> None:
    downstream = InMemoryEventSink()
    sink = BoundedEventSink(downstream, EventDeliveryPolicy(max_capacity=32))
    for label in ("A", "B", "C"):
        sink.publish(
            _event(label, seq=ord(label)),
            priority=EventPriority.CRITICAL,
        )
    sink.close()
    kinds = [ev.kind for _prio, ev in downstream.records]
    assert kinds == ["A", "B", "C"]


def test_cancellation_during_publish_no_orphan_worker() -> None:
    downstream = InMemoryEventSink(consume_delay_seconds=0.05)
    sink = BoundedEventSink(downstream, EventDeliveryPolicy(max_capacity=8))
    errors: list[BaseException] = []

    def _publisher() -> None:
        try:
            for i in range(50):
                sink.publish(
                    DeliverableEvent(
                        event_id=f"pub-{i}",
                        kind=CriticalEventKind.RECOVERY_STATE_CHANGE.value,
                        sequence=i,
                    ),
                    priority=EventPriority.CRITICAL,
                )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=_publisher)
    worker.start()
    time.sleep(0.05)
    sink.close()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert downstream.records
