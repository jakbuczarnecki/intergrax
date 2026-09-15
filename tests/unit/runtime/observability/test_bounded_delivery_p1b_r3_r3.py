# © Artur Czarnecki. All rights reserved.

"""P1B-R3-R3 — shutdown terminal-state race: normal drain vs worker death during close."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from intergrax.contracts.event_delivery import (
    DeliverableEvent,
    EventDeliveryBoundaryError,
    EventDeliveryBoundaryFailureKind,
    EventDeliveryDisposition,
    EventDeliveryObligation,
    EventDeliveryPolicy,
    EventDeliveryResult,
    EventPriority,
    EventSinkHealthState,
    make_deliverable_event,
    make_observability_export_payload,
)
from intergrax.runtime.observability.event_delivery import BoundedEventSink

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PAYLOAD = make_observability_export_payload(event_id="ev-r3", kind="DECISION_FINALIZED")
_DELIVERABLE = make_deliverable_event(_PAYLOAD)


def _accept_downstream():
    class _AcceptDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.ADMISSION,
            )

        def close(self) -> None:
            return None

    return _AcceptDownstream()


@dataclass
class _BlockedBacklogHarness:
    bounded: BoundedEventSink
    block_publish: threading.Event
    crash_on_next_item: threading.Event


def _blocked_backlog_harness(*, backlog_event_id: str) -> _BlockedBacklogHarness:
    entered_publish = threading.Event()
    block_publish = threading.Event()
    crash_on_next_item = threading.Event()

    class _BlockingDownstream:
        def publish(self, event: DeliverableEvent, *, priority, deadline=None) -> EventDeliveryResult:
            entered_publish.set()
            block_publish.wait(timeout=5.0)
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.ADMISSION,
            )

        def close(self) -> None:
            block_publish.set()

    policy = EventDeliveryPolicy(max_capacity=4, drain_shutdown_timeout_seconds=2.0)
    bounded = BoundedEventSink(_BlockingDownstream(), policy)
    original_process = bounded._process_item

    def _crash_on_draining_backlog(item) -> None:
        if item is not None and crash_on_next_item.is_set():
            raise RuntimeError("controlled worker death during close drain")
        original_process(item)

    bounded._process_item = _crash_on_draining_backlog  # noqa: SLF001

    assert (
        bounded.publish(_DELIVERABLE, priority=EventPriority.IMPORTANT).disposition
        is EventDeliveryDisposition.ACCEPTED
    )
    assert entered_publish.wait(timeout=2.0)
    backlog = make_deliverable_event(
        make_observability_export_payload(event_id=backlog_event_id, kind="DECISION_FINALIZED"),
    )
    assert (
        bounded.publish(backlog, priority=EventPriority.IMPORTANT).disposition
        is EventDeliveryDisposition.ACCEPTED
    )
    return _BlockedBacklogHarness(
        bounded=bounded,
        block_publish=block_publish,
        crash_on_next_item=crash_on_next_item,
    )


def _close_with_worker_crash(harness: _BlockedBacklogHarness) -> list[EventDeliveryBoundaryError]:
    close_error: list[EventDeliveryBoundaryError] = []
    close_started = threading.Event()

    def _close() -> None:
        close_started.set()
        try:
            harness.bounded.close()
        except EventDeliveryBoundaryError as exc:
            close_error.append(exc)

    closer = threading.Thread(target=_close)
    closer.start()
    assert close_started.wait(timeout=2.0)
    harness.crash_on_next_item.set()
    harness.block_publish.set()
    closer.join(timeout=5.0)
    return close_error


def test_normal_close_confirms_drained_normally_and_healthy() -> None:
    policy = EventDeliveryPolicy(max_capacity=4, drain_shutdown_timeout_seconds=2.0)
    bounded = BoundedEventSink(_accept_downstream(), policy)
    assert bounded._worker.is_alive()  # noqa: SLF001
    bounded.close()
    assert bounded._worker_drained_normally.is_set()  # noqa: SLF001
    assert bounded.health_state() is EventSinkHealthState.HEALTHY
    assert not bounded._worker.is_alive()  # noqa: SLF001


def test_worker_dies_during_close_before_normal_drain_raises_and_unhealthy() -> None:
    harness = _blocked_backlog_harness(backlog_event_id="ev-r3-backlog")
    close_error = _close_with_worker_crash(harness)
    bounded = harness.bounded
    assert len(close_error) == 1
    assert close_error[0].kind is EventDeliveryBoundaryFailureKind.SINK_UNAVAILABLE
    assert bounded.health_state() is EventSinkHealthState.UNHEALTHY
    assert not bounded._worker_drained_normally.is_set()  # noqa: SLF001


def test_repeat_close_after_terminal_shutdown_failure_still_raises() -> None:
    harness = _blocked_backlog_harness(backlog_event_id="ev-r3-backlog2")
    close_error = _close_with_worker_crash(harness)
    bounded = harness.bounded
    assert len(close_error) == 1
    assert bounded.health_state() is EventSinkHealthState.UNHEALTHY
    with pytest.raises(EventDeliveryBoundaryError) as raised:
        bounded.close()
    assert raised.value.kind is EventDeliveryBoundaryFailureKind.SINK_UNAVAILABLE


def test_publish_after_terminal_shutdown_failure_not_accepted() -> None:
    harness = _blocked_backlog_harness(backlog_event_id="ev-r3-backlog3")
    close_error = _close_with_worker_crash(harness)
    bounded = harness.bounded
    assert len(close_error) == 1
    result = bounded.publish(_DELIVERABLE, priority=EventPriority.BEST_EFFORT)
    assert result.disposition is EventDeliveryDisposition.REJECTED
    assert result.disposition is not EventDeliveryDisposition.ACCEPTED


def test_worker_dies_during_sentinel_enqueue_while_queue_full_raises() -> None:
    entered_publish = threading.Event()
    block_publish = threading.Event()
    crash_worker = threading.Event()

    class _BlockingDownstream:
        def publish(self, event: DeliverableEvent, *, priority, deadline=None) -> EventDeliveryResult:
            entered_publish.set()
            block_publish.wait(timeout=5.0)
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.ADMISSION,
            )

        def close(self) -> None:
            block_publish.set()

    policy = EventDeliveryPolicy(max_capacity=2, drain_shutdown_timeout_seconds=2.0)
    bounded = BoundedEventSink(_BlockingDownstream(), policy)
    original_process = bounded._process_item

    def _process_with_crash(item) -> None:
        if crash_worker.is_set() and item is not None:
            raise RuntimeError("controlled worker death during sentinel enqueue wait")
        original_process(item)

    bounded._process_item = _process_with_crash  # noqa: SLF001

    assert (
        bounded.publish(_DELIVERABLE, priority=EventPriority.IMPORTANT).disposition
        is EventDeliveryDisposition.ACCEPTED
    )
    assert entered_publish.wait(timeout=2.0)
    for event_id in ("ev-r3-b1", "ev-r3-b2"):
        backlog = make_deliverable_event(
            make_observability_export_payload(event_id=event_id, kind="DECISION_FINALIZED"),
        )
        assert (
            bounded.publish(backlog, priority=EventPriority.IMPORTANT).disposition
            is EventDeliveryDisposition.ACCEPTED
        )

    close_error = _close_with_worker_crash(
        _BlockedBacklogHarness(
            bounded=bounded,
            block_publish=block_publish,
            crash_on_next_item=crash_worker,
        ),
    )
    assert len(close_error) == 1
    assert close_error[0].kind is EventDeliveryBoundaryFailureKind.SINK_UNAVAILABLE
    assert bounded.health_state() is EventSinkHealthState.UNHEALTHY
    assert not bounded._worker_drained_normally.is_set()  # noqa: SLF001


def test_worker_dies_before_close_attempt_raises_on_close() -> None:
    entered_process = threading.Event()
    block_process = threading.Event()

    class _AcceptDownstream:
        def publish(self, event: DeliverableEvent, *, priority, deadline=None) -> EventDeliveryResult:
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.ADMISSION,
            )

        def close(self) -> None:
            return None

    policy = EventDeliveryPolicy(max_capacity=1, drain_shutdown_timeout_seconds=2.0)
    bounded = BoundedEventSink(_AcceptDownstream(), policy)
    original_process = bounded._process_item

    def _crash_after_block(item) -> None:
        if item is not None:
            entered_process.set()
            block_process.wait(timeout=5.0)
            raise RuntimeError("controlled worker death before close")
        original_process(item)

    bounded._process_item = _crash_after_block  # noqa: SLF001
    assert (
        bounded.publish(_DELIVERABLE, priority=EventPriority.IMPORTANT).disposition
        is EventDeliveryDisposition.ACCEPTED
    )
    assert entered_process.wait(timeout=2.0)
    block_process.set()
    bounded._worker.join(timeout=2.0)  # noqa: SLF001
    assert not bounded._worker.is_alive()  # noqa: SLF001
    assert bounded.health_state() is EventSinkHealthState.UNHEALTHY

    with pytest.raises(EventDeliveryBoundaryError) as raised:
        bounded.close()
    assert raised.value.kind is EventDeliveryBoundaryFailureKind.SINK_UNAVAILABLE
