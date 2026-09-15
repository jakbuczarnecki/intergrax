# © Artur Czarnecki. All rights reserved.

"""W5-A — observability event delivery backpressure qualification."""

from __future__ import annotations

import threading
import time

import pytest

from intergrax.contracts.event_delivery import (
    CriticalEventKind,
    DeliverableEvent,
    EventDeliveryDisposition,
    EventDeliveryObligation,
    EventDeliveryPolicy,
    EventDeliveryResult,
    EventPriority,
    make_deliverable_event,
    make_observability_export_payload,
)
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    InMemoryEventSink,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _event(label: str, seq: int = 0) -> DeliverableEvent:
    return make_deliverable_event(
        make_observability_export_payload(event_id=label, kind=label),
        sequence=seq,
    )


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
    worker_entered_downstream = threading.Event()
    release_downstream = threading.Event()
    downstream_deliveries: list[str] = []
    deliveries_lock = threading.Lock()

    class _PhasedGatedDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            worker_entered_downstream.set()
            if not release_downstream.wait(timeout=10.0):
                raise TimeoutError("downstream release timed out")
            with deliveries_lock:
                downstream_deliveries.append(event.event_id)
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.COMPLETION,
            )

        def close(self) -> None:
            release_downstream.set()

    policy = EventDeliveryPolicy(max_capacity=10)
    sink = BoundedEventSink(_PhasedGatedDownstream(), policy)
    errors: list[BaseException] = []
    sync_timeout = 10.0

    def _first_in_flight() -> None:
        try:
            sink.publish(_event("c-in-flight"), priority=EventPriority.CRITICAL)
        except BaseException as exc:
            errors.append(exc)

    first_publisher = threading.Thread(
        target=_first_in_flight, name="w5a-first-critical"
    )
    first_publisher.start()
    assert worker_entered_downstream.wait(timeout=sync_timeout)

    fill_barrier = threading.Barrier(policy.max_capacity)

    def _fill_queue_slot(label: str) -> None:
        try:
            fill_barrier.wait(timeout=sync_timeout)
            sink.publish(_event(label), priority=EventPriority.CRITICAL)
        except BaseException as exc:
            errors.append(exc)

    fill_publishers = [
        threading.Thread(
            target=_fill_queue_slot,
            args=(f"c-fill-{index}",),
            name=f"w5a-fill-{index}",
        )
        for index in range(policy.max_capacity)
    ]
    for thread in fill_publishers:
        thread.start()

    depth_deadline = time.monotonic() + sync_timeout
    while sink.pending_depth < policy.max_capacity:
        if time.monotonic() >= depth_deadline:
            pytest.fail(
                f"queue never reached capacity: pending_depth={sink.pending_depth} "
                f"max_capacity={policy.max_capacity}",
            )

    overflow_result = sink.publish(
        _event("c-overflow"), priority=EventPriority.CRITICAL
    )
    assert overflow_result.disposition is EventDeliveryDisposition.REJECTED
    assert "c-overflow" not in downstream_deliveries

    release_downstream.set()
    first_publisher.join(timeout=sync_timeout)
    for thread in fill_publishers:
        thread.join(timeout=sync_timeout)

    assert errors == []
    assert not first_publisher.is_alive()
    for thread in fill_publishers:
        assert not thread.is_alive()

    sink.close()
    assert len(downstream_deliveries) == policy.max_capacity + 1


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
                    make_deliverable_event(
                        make_observability_export_payload(
                            event_id=f"pub-{i}",
                            kind=CriticalEventKind.RECOVERY_STATE_CHANGE.value,
                        ),
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
