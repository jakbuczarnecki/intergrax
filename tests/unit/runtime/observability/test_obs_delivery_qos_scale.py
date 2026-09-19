# © Artur Czarnecki. All rights reserved.

"""OBS-DELIVERY-QOS-SCALE — bounded priority delivery qualification."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar

import pytest

from intergrax.contracts.event_delivery import (
    DeliverableEvent,
    EventDeliveryAdmissionPolicyPort,
    EventDeliveryBoundaryError,
    EventDeliveryBufferEntry,
    EventDeliveryDisposition,
    EventDeliveryObligation,
    EventDeliveryPolicy,
    EventDeliveryResult,
    EventPriority,
    EventSinkHealthState,
    EventSinkPort,
    make_deliverable_event,
    make_observability_export_payload,
)
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    InMemoryEventSink,
)
from intergrax.runtime.observability.event_delivery.enterprise_default_event_delivery_admission_policy import (
    EnterpriseDefaultEventDeliveryAdmissionPolicy,
)
from intergrax.runtime.observability.event_delivery.queue_backed_event_delivery_buffer import (
    QueueBackedEventDeliveryBuffer,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SYNC_TIMEOUT = 10.0
_TBuffered = TypeVar("_TBuffered")


def _event(label: str) -> DeliverableEvent:
    return make_deliverable_event(
        make_observability_export_payload(event_id=label, kind=label),
    )


def test_baseline_capacity_starvation_critical_rejected_without_reserve() -> None:
    """Lower-priority traffic can consume the full buffer when reserve is zero."""
    max_capacity = 4
    policy = EventDeliveryPolicy(
        max_capacity=max_capacity, critical_reserved_capacity=0
    )
    downstream = InMemoryEventSink()
    sink = BoundedEventSink(downstream, policy)
    for index in range(max_capacity):
        result = sink.publish(_event(f"be-{index}"), priority=EventPriority.BEST_EFFORT)
        assert result.disposition is EventDeliveryDisposition.ACCEPTED
    overflow = sink.publish(_event("critical-late"), priority=EventPriority.CRITICAL)
    assert overflow.disposition is EventDeliveryDisposition.REJECTED
    sink.close()


def test_critical_reserved_admission_under_best_effort_pressure() -> None:
    max_capacity = 10
    reserve = 2
    policy = EventDeliveryPolicy(
        max_capacity=max_capacity,
        critical_reserved_capacity=reserve,
        critical_completion_timeout_seconds=30.0,
    )
    worker_in_downstream = threading.Event()
    release_downstream = threading.Event()
    downstream_ids: list[str] = []
    downstream_lock = threading.Lock()

    class _GatedDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            worker_in_downstream.set()
            if not release_downstream.wait(timeout=_SYNC_TIMEOUT):
                raise TimeoutError("release timed out")
            with downstream_lock:
                downstream_ids.append(event.event_id)
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.COMPLETION,
            )

        def close(self) -> None:
            release_downstream.set()

    sink = BoundedEventSink(_GatedDownstream(), policy)
    holder = threading.Thread(
        target=lambda: sink.publish(_event("c-hold"), priority=EventPriority.CRITICAL),
        name="c-hold",
    )
    holder.start()
    assert worker_in_downstream.wait(timeout=_SYNC_TIMEOUT)

    non_critical_limit = max_capacity - reserve
    fill_barrier = threading.Barrier(non_critical_limit)

    def _fill(label: str) -> None:
        fill_barrier.wait(timeout=_SYNC_TIMEOUT)
        sink.publish(_event(label), priority=EventPriority.BEST_EFFORT)

    fillers = [
        threading.Thread(target=_fill, args=(f"be-{index}",), name=f"be-fill-{index}")
        for index in range(non_critical_limit)
    ]
    for thread in fillers:
        thread.start()
    depth_deadline = time.monotonic() + _SYNC_TIMEOUT
    while sink.pending_depth < non_critical_limit:
        if time.monotonic() >= depth_deadline:
            pytest.fail(f"non-critical buffer not filled: depth={sink.pending_depth}")

    critical_results: list[EventDeliveryResult] = []

    def _publish_critical() -> None:
        critical_results.append(
            sink.publish(_event("critical-admit"), priority=EventPriority.CRITICAL),
        )

    critical_publisher = threading.Thread(target=_publish_critical, name="c-admit")
    critical_publisher.start()
    admit_deadline = time.monotonic() + _SYNC_TIMEOUT
    while sink.pending_depth < non_critical_limit + 1:
        if time.monotonic() >= admit_deadline:
            pytest.fail("CRITICAL was not admitted into the bounded buffer")
        time.sleep(0.001)

    release_downstream.set()
    critical_publisher.join(timeout=_SYNC_TIMEOUT)
    holder.join(timeout=_SYNC_TIMEOUT)
    for thread in fillers:
        thread.join(timeout=_SYNC_TIMEOUT)
    sink.close()
    assert critical_results
    assert critical_results[0].disposition is EventDeliveryDisposition.ACCEPTED
    assert "critical-admit" in downstream_ids


def test_hol_queued_critical_waits_behind_lower_priority_backlog() -> None:
    """Queue HOL: admitted CRITICAL is still drained FIFO behind earlier BE items."""
    worker_in_downstream = threading.Event()
    release_downstream = threading.Event()
    downstream_order: list[str] = []
    order_lock = threading.Lock()

    class _SingleFlightDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            worker_in_downstream.set()
            if not release_downstream.wait(timeout=_SYNC_TIMEOUT):
                raise TimeoutError("release timed out")
            with order_lock:
                downstream_order.append(event.event_id)
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.COMPLETION,
            )

        def close(self) -> None:
            release_downstream.set()

    policy = EventDeliveryPolicy(
        max_capacity=8,
        critical_reserved_capacity=2,
        critical_completion_timeout_seconds=60.0,
    )
    sink = BoundedEventSink(_SingleFlightDownstream(), policy)

    first_critical = threading.Thread(
        target=lambda: sink.publish(_event("c-first"), priority=EventPriority.CRITICAL),
        name="c-first",
    )
    first_critical.start()
    assert worker_in_downstream.wait(timeout=_SYNC_TIMEOUT)

    for index in range(3):
        sink.publish(_event(f"be-{index}"), priority=EventPriority.BEST_EFFORT)

    late_results: list[EventDeliveryResult] = []

    def _publish_late() -> None:
        late_results.append(
            sink.publish(_event("c-late"), priority=EventPriority.CRITICAL)
        )

    late_publisher = threading.Thread(target=_publish_late, name="c-late")
    late_publisher.start()
    admit_deadline = time.monotonic() + _SYNC_TIMEOUT
    while sink.pending_depth < 4:
        if time.monotonic() >= admit_deadline:
            pytest.fail("late CRITICAL was not admitted behind backlog")
        time.sleep(0.001)

    release_downstream.set()
    late_publisher.join(timeout=_SYNC_TIMEOUT)
    first_critical.join(timeout=_SYNC_TIMEOUT)
    sink.close()

    assert late_results
    assert late_results[0].disposition is EventDeliveryDisposition.ACCEPTED
    assert downstream_order[0] == "c-first"
    assert downstream_order.index("c-late") > downstream_order.index("be-0")


def test_important_deferred_when_non_critical_capacity_exhausted() -> None:
    max_capacity = 4
    reserve = 1
    policy = EventDeliveryPolicy(
        max_capacity=max_capacity,
        critical_reserved_capacity=reserve,
        important_wait_timeout_seconds=0.15,
    )
    downstream, entered, release = _blocking_first_downstream()
    sink = BoundedEventSink(downstream, policy)
    non_critical_limit = max_capacity - reserve
    _fill_non_critical_buffered(
        sink,
        target=non_critical_limit,
        downstream_entered=entered,
        label="defer",
    )
    deferred = sink.publish(_event("important"), priority=EventPriority.IMPORTANT)
    assert deferred.disposition is EventDeliveryDisposition.DEFERRED
    release.set()
    sink.close()


def test_best_effort_drop_does_not_affect_queue_critical_slots() -> None:
    policy = EventDeliveryPolicy(max_capacity=5, critical_reserved_capacity=2)
    sink = BoundedEventSink(InMemoryEventSink(), policy)
    for index in range(10):
        sink.publish(_event(f"be-{index}"), priority=EventPriority.BEST_EFFORT)
    assert sink.pending_depth <= policy.max_capacity - policy.critical_reserved_capacity
    critical = sink.publish(_event("c"), priority=EventPriority.CRITICAL)
    assert critical.disposition is EventDeliveryDisposition.ACCEPTED
    sink.close()


def test_custom_admission_policy_without_subclassing_enterprise_default() -> None:
    class _NoNonCriticalBuffer:
        def max_non_critical_buffered_events(self, policy: EventDeliveryPolicy) -> int:
            return 0

    policy = EventDeliveryPolicy(max_capacity=4, critical_reserved_capacity=0)
    custom: EventDeliveryAdmissionPolicyPort = _NoNonCriticalBuffer()
    sink = BoundedEventSink(
        InMemoryEventSink(),
        policy,
        admission_policy=custom,
    )
    dropped = sink.publish(_event("be"), priority=EventPriority.BEST_EFFORT)
    assert dropped.disposition is EventDeliveryDisposition.DROPPED
    critical = sink.publish(_event("c"), priority=EventPriority.CRITICAL)
    assert critical.disposition is EventDeliveryDisposition.ACCEPTED
    sink.close()


def test_bounded_memory_never_exceeds_configured_capacity() -> None:
    policy = EventDeliveryPolicy(max_capacity=6, critical_reserved_capacity=2)
    sink = BoundedEventSink(InMemoryEventSink(consume_delay_seconds=0.01), policy)
    peak = 0
    for index in range(200):
        sink.publish(_event(f"mix-{index}"), priority=EventPriority.BEST_EFFORT)
        peak = max(peak, sink.pending_depth)
        if index % 7 == 0:
            sink.publish(_event(f"c-{index}"), priority=EventPriority.CRITICAL)
        peak = max(peak, sink.pending_depth)
    assert peak <= policy.max_capacity
    sink.close()


def test_default_admission_policy_matches_enterprise_reserve_field() -> None:
    policy = EventDeliveryPolicy(max_capacity=20, critical_reserved_capacity=3)
    admission = EnterpriseDefaultEventDeliveryAdmissionPolicy()
    assert admission.max_non_critical_buffered_events(policy) == 17


def _blocking_first_downstream() -> tuple[
    EventSinkPort, threading.Event, threading.Event
]:
    """Block downstream until ``release``; only the first in-flight publish holds the gate."""
    entered = threading.Event()
    release = threading.Event()
    gate_taken = threading.Event()

    class _Downstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            entered.set()
            if not gate_taken.is_set():
                gate_taken.set()
                if not release.wait(timeout=_SYNC_TIMEOUT):
                    raise TimeoutError("release timed out")
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.ADMISSION,
            )

        def close(self) -> None:
            release.set()

    return _Downstream(), entered, release


def _prime_blocked_downstream(sink: BoundedEventSink, label: str) -> None:
    result = sink.publish(
        _event(f"{label}-prime"),
        priority=EventPriority.BEST_EFFORT,
    )
    assert result.disposition is EventDeliveryDisposition.ACCEPTED


def _fill_non_critical_buffered(
    sink: BoundedEventSink,
    *,
    target: int,
    downstream_entered: threading.Event,
    label: str,
) -> None:
    _prime_blocked_downstream(sink, label)
    assert downstream_entered.wait(timeout=_SYNC_TIMEOUT)
    for index in range(target):
        result = sink.publish(
            _event(f"{label}-be-{index}"),
            priority=EventPriority.BEST_EFFORT,
        )
        assert result.disposition is EventDeliveryDisposition.ACCEPTED


def test_r1_concurrent_important_respects_non_critical_peak() -> None:
    max_capacity = 10
    reserve = 2
    non_critical_limit = max_capacity - reserve
    policy = EventDeliveryPolicy(
        max_capacity=max_capacity,
        critical_reserved_capacity=reserve,
        important_wait_timeout_seconds=2.0,
    )
    downstream, entered, release = _blocking_first_downstream()
    sink = BoundedEventSink(downstream, policy)
    _fill_non_critical_buffered(
        sink,
        target=non_critical_limit,
        downstream_entered=entered,
        label="pre",
    )
    peak_holder = {"depth": sink.pending_depth}
    barrier = threading.Barrier(20)
    results: list[EventDeliveryResult] = []
    results_lock = threading.Lock()

    def _publish_important(index: int) -> None:
        barrier.wait(timeout=_SYNC_TIMEOUT)
        result = sink.publish(
            _event(f"imp-{index}"),
            priority=EventPriority.IMPORTANT,
        )
        with results_lock:
            results.append(result)
            peak_holder["depth"] = max(peak_holder["depth"], sink.pending_depth)

    threads = [
        threading.Thread(target=_publish_important, args=(index,), name=f"imp-{index}")
        for index in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_SYNC_TIMEOUT + 2.0)
    release.set()
    sink.close()
    accepted = [
        result
        for result in results
        if result.disposition is EventDeliveryDisposition.ACCEPTED
    ]
    assert peak_holder["depth"] <= non_critical_limit
    assert len(accepted) <= 1


def test_r1_mixed_best_effort_and_important_race() -> None:
    max_capacity = 8
    reserve = 2
    non_critical_limit = max_capacity - reserve
    policy = EventDeliveryPolicy(
        max_capacity=max_capacity,
        critical_reserved_capacity=reserve,
        important_wait_timeout_seconds=2.0,
    )
    downstream, entered, release = _blocking_first_downstream()
    sink = BoundedEventSink(downstream, policy)
    _fill_non_critical_buffered(
        sink,
        target=non_critical_limit - 1,
        downstream_entered=entered,
        label="mix",
    )
    peak_holder = {"depth": sink.pending_depth}
    barrier = threading.Barrier(30)
    results_lock = threading.Lock()
    results: list[EventDeliveryResult] = []

    def _publish_mixed(index: int) -> None:
        barrier.wait(timeout=_SYNC_TIMEOUT)
        priority = (
            EventPriority.BEST_EFFORT if index % 2 == 0 else EventPriority.IMPORTANT
        )
        result = sink.publish(_event(f"m-{index}"), priority=priority)
        with results_lock:
            results.append(result)
            peak_holder["depth"] = max(peak_holder["depth"], sink.pending_depth)

    threads = [
        threading.Thread(target=_publish_mixed, args=(index,), name=f"mix-{index}")
        for index in range(30)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_SYNC_TIMEOUT + 2.0)
    release.set()
    sink.close()
    assert peak_holder["depth"] <= non_critical_limit


def test_r1_custom_quota_one_concurrent_publishers() -> None:
    class _SingleSlot:
        def max_non_critical_buffered_events(self, policy: EventDeliveryPolicy) -> int:
            return 1

    policy = EventDeliveryPolicy(max_capacity=4, critical_reserved_capacity=0)
    downstream, entered, release = _blocking_first_downstream()
    sink = BoundedEventSink(
        downstream,
        policy,
        admission_policy=_SingleSlot(),
    )
    sink.publish(_event("hold"), priority=EventPriority.BEST_EFFORT)
    assert entered.wait(timeout=_SYNC_TIMEOUT)
    peak_holder = {"depth": sink.pending_depth}
    barrier = threading.Barrier(10)
    results: list[EventDeliveryResult] = []
    lock = threading.Lock()

    def _be(index: int) -> None:
        barrier.wait(timeout=_SYNC_TIMEOUT)
        result = sink.publish(_event(f"be-{index}"), priority=EventPriority.BEST_EFFORT)
        with lock:
            results.append(result)
            peak_holder["depth"] = max(peak_holder["depth"], sink.pending_depth)

    threads = [threading.Thread(target=_be, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_SYNC_TIMEOUT)
    release.set()
    sink.close()
    assert peak_holder["depth"] <= 1


@pytest.mark.parametrize(
    ("return_value", "error_type"),
    [
        (-1, ValueError),
        (True, TypeError),
        (11, ValueError),
    ],
)
def test_r1_invalid_custom_admission_limit_rejected(
    return_value: object,
    error_type: type[BaseException],
) -> None:
    class _BadPolicy:
        def max_non_critical_buffered_events(self, policy: EventDeliveryPolicy) -> int:
            return return_value  # type: ignore[return-value]

    policy = EventDeliveryPolicy(max_capacity=10, critical_reserved_capacity=2)
    bad_policy: EventDeliveryAdmissionPolicyPort = _BadPolicy()
    with pytest.raises(error_type):
        BoundedEventSink(InMemoryEventSink(), policy, admission_policy=bad_policy)


def test_r1_important_waits_then_accepts_when_quota_released() -> None:
    policy = EventDeliveryPolicy(
        max_capacity=6,
        critical_reserved_capacity=1,
        important_wait_timeout_seconds=2.0,
    )
    downstream, entered, release = _blocking_first_downstream()
    sink = BoundedEventSink(downstream, policy)
    non_critical_limit = policy.max_capacity - policy.critical_reserved_capacity
    _fill_non_critical_buffered(
        sink,
        target=non_critical_limit,
        downstream_entered=entered,
        label="wait",
    )
    important_done = threading.Event()
    important_result: list[EventDeliveryResult] = []

    def _important() -> None:
        important_result.append(
            sink.publish(_event("important"), priority=EventPriority.IMPORTANT),
        )
        important_done.set()

    thread = threading.Thread(target=_important, name="important-wait")
    thread.start()
    time.sleep(0.2)
    assert thread.is_alive()
    release.set()
    thread.join(timeout=_SYNC_TIMEOUT)
    assert important_done.is_set()
    assert important_result[0].disposition is EventDeliveryDisposition.ACCEPTED
    sink.close()


def test_r1_important_defers_after_bounded_wait() -> None:
    policy = EventDeliveryPolicy(
        max_capacity=4,
        critical_reserved_capacity=1,
        important_wait_timeout_seconds=0.15,
    )
    downstream, entered, release = _blocking_first_downstream()
    sink = BoundedEventSink(downstream, policy)
    non_critical_limit = policy.max_capacity - policy.critical_reserved_capacity
    _fill_non_critical_buffered(
        sink,
        target=non_critical_limit,
        downstream_entered=entered,
        label="timeout",
    )
    started = time.monotonic()
    result = sink.publish(_event("late"), priority=EventPriority.IMPORTANT)
    elapsed = time.monotonic() - started
    assert result.disposition is EventDeliveryDisposition.DEFERRED
    assert elapsed >= 0.1
    assert elapsed < _SYNC_TIMEOUT
    release.set()
    sink.close()


def test_r1_close_while_important_waits_on_quota() -> None:
    policy = EventDeliveryPolicy(
        max_capacity=4,
        critical_reserved_capacity=1,
        important_wait_timeout_seconds=5.0,
    )
    downstream, entered, release = _blocking_first_downstream()
    sink = BoundedEventSink(downstream, policy)
    non_critical_limit = policy.max_capacity - policy.critical_reserved_capacity
    _fill_non_critical_buffered(
        sink,
        target=non_critical_limit,
        downstream_entered=entered,
        label="close",
    )
    results: list[EventDeliveryResult] = []

    def _important() -> None:
        results.append(
            sink.publish(_event("waiting"), priority=EventPriority.IMPORTANT),
        )

    thread = threading.Thread(target=_important, name="close-wait")
    thread.start()
    time.sleep(0.05)
    sink.close()
    thread.join(timeout=_SYNC_TIMEOUT)
    assert results
    assert results[0].disposition is EventDeliveryDisposition.REJECTED
    release.set()


def test_r1_fill_drain_cycles_do_not_leak_quota() -> None:
    policy = EventDeliveryPolicy(max_capacity=6, critical_reserved_capacity=2)
    non_critical_limit = policy.max_capacity - policy.critical_reserved_capacity
    for _cycle in range(3):
        sink = BoundedEventSink(InMemoryEventSink(consume_delay_seconds=0.0), policy)
        for index in range(non_critical_limit):
            result = sink.publish(
                _event(f"c-{_cycle}-{index}"),
                priority=EventPriority.BEST_EFFORT,
            )
            assert result.disposition is EventDeliveryDisposition.ACCEPTED
        deadline = time.monotonic() + _SYNC_TIMEOUT
        while sink.pending_depth > 0:
            if time.monotonic() >= deadline:
                sink.close()
                pytest.fail("drain did not complete")
            time.sleep(0.001)
        accepted = sink.publish(
            _event(f"post-{_cycle}"),
            priority=EventPriority.BEST_EFFORT,
        )
        assert accepted.disposition is EventDeliveryDisposition.ACCEPTED
        sink.close()


def test_r1_custom_quota_zero_important_waits_then_defers() -> None:
    class _ZeroQuota:
        def max_non_critical_buffered_events(self, policy: EventDeliveryPolicy) -> int:
            return 0

    policy = EventDeliveryPolicy(
        max_capacity=4,
        critical_reserved_capacity=0,
        important_wait_timeout_seconds=0.1,
    )
    sink = BoundedEventSink(
        InMemoryEventSink(),
        policy,
        admission_policy=_ZeroQuota(),
    )
    dropped = sink.publish(_event("be"), priority=EventPriority.BEST_EFFORT)
    assert dropped.disposition is EventDeliveryDisposition.DROPPED
    deferred = sink.publish(_event("imp"), priority=EventPriority.IMPORTANT)
    assert deferred.disposition is EventDeliveryDisposition.DEFERRED
    critical = sink.publish(_event("c"), priority=EventPriority.CRITICAL)
    assert critical.disposition is EventDeliveryDisposition.ACCEPTED
    sink.close()


def test_r1_physical_queue_full_blocks_important_despite_quota_headroom() -> None:
    max_capacity = 3
    policy = EventDeliveryPolicy(
        max_capacity=max_capacity,
        critical_reserved_capacity=0,
        important_wait_timeout_seconds=0.2,
    )
    downstream, entered, release = _blocking_first_downstream()
    sink = BoundedEventSink(downstream, policy)
    _fill_non_critical_buffered(
        sink,
        target=max_capacity,
        downstream_entered=entered,
        label="phys",
    )
    assert sink.pending_depth == max_capacity
    result = sink.publish(_event("important"), priority=EventPriority.IMPORTANT)
    assert result.disposition is EventDeliveryDisposition.DEFERRED
    release.set()
    sink.close()


def test_r2_shutdown_wins_when_quota_freed_after_stop() -> None:
    """Quota release after shutdown must not grant a new non-critical reservation."""
    policy = EventDeliveryPolicy(
        max_capacity=4,
        critical_reserved_capacity=1,
        important_wait_timeout_seconds=5.0,
    )
    downstream, entered, release = _blocking_first_downstream()
    sink = BoundedEventSink(downstream, policy)
    non_critical_limit = policy.max_capacity - policy.critical_reserved_capacity
    _fill_non_critical_buffered(
        sink,
        target=non_critical_limit,
        downstream_entered=entered,
        label="r2-stop",
    )
    results: list[EventDeliveryResult] = []
    important_done = threading.Event()

    def _important() -> None:
        results.append(
            sink.publish(_event("r2-waiting"), priority=EventPriority.IMPORTANT),
        )
        important_done.set()

    thread = threading.Thread(target=_important, name="r2-important")
    thread.start()
    thread.join(timeout=0.25)
    assert not important_done.is_set(), "important should block on quota"
    sink.close()
    release.set()
    assert important_done.wait(timeout=_SYNC_TIMEOUT)
    assert results
    assert results[0].disposition is EventDeliveryDisposition.REJECTED
    post = sink.publish(_event("post-close"), priority=EventPriority.IMPORTANT)
    assert post.disposition is EventDeliveryDisposition.REJECTED


def test_r2_admission_wins_before_shutdown_drains() -> None:
    policy = EventDeliveryPolicy(
        max_capacity=4,
        critical_reserved_capacity=1,
        important_wait_timeout_seconds=5.0,
    )
    recorded: list[str] = []
    record_lock = threading.Lock()
    entered = threading.Event()
    release = threading.Event()
    gate_taken = threading.Event()

    class _RecordingDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            entered.set()
            if not gate_taken.is_set():
                gate_taken.set()
                if not release.wait(timeout=_SYNC_TIMEOUT):
                    raise TimeoutError("release timed out")
            with record_lock:
                recorded.append(event.event_id)
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.ADMISSION,
            )

        def close(self) -> None:
            release.set()

    sink = BoundedEventSink(_RecordingDownstream(), policy)
    non_critical_limit = policy.max_capacity - policy.critical_reserved_capacity
    _fill_non_critical_buffered(
        sink,
        target=non_critical_limit,
        downstream_entered=entered,
        label="r2-win",
    )
    admitted = threading.Event()
    results: list[EventDeliveryResult] = []

    def _important() -> None:
        results.append(
            sink.publish(_event("r2-admitted"), priority=EventPriority.IMPORTANT),
        )
        admitted.set()

    thread = threading.Thread(target=_important, name="r2-admit")
    thread.start()
    release.set()
    assert admitted.wait(timeout=_SYNC_TIMEOUT)
    assert results[0].disposition is EventDeliveryDisposition.ACCEPTED
    sink.close()
    assert "r2-admitted" in recorded


def test_r2_best_effort_rejected_after_close_linearization() -> None:
    policy = EventDeliveryPolicy(max_capacity=4, critical_reserved_capacity=0)
    sink = BoundedEventSink(InMemoryEventSink(), policy)
    sink.close()
    result = sink.publish(_event("be-late"), priority=EventPriority.BEST_EFFORT)
    assert result.disposition is EventDeliveryDisposition.REJECTED


def test_r2_critical_rejected_after_close_linearization() -> None:
    policy = EventDeliveryPolicy(
        max_capacity=4,
        critical_reserved_capacity=1,
        critical_completion_timeout_seconds=5.0,
    )
    sink = BoundedEventSink(InMemoryEventSink(), policy)
    sink.close()
    result = sink.publish(_event("c-late"), priority=EventPriority.CRITICAL)
    assert result.disposition is EventDeliveryDisposition.REJECTED


def test_r2_worker_death_rejects_quota_waiter() -> None:
    policy = EventDeliveryPolicy(
        max_capacity=4,
        critical_reserved_capacity=1,
        important_wait_timeout_seconds=5.0,
    )
    downstream, entered, release = _blocking_first_downstream()
    buffer = _WorkerDeathEventDeliveryBuffer(capacity=policy.max_capacity)
    sink = BoundedEventSink(downstream, policy, buffer=buffer)
    non_critical_limit = policy.max_capacity - policy.critical_reserved_capacity
    _fill_non_critical_buffered(
        sink,
        target=non_critical_limit,
        downstream_entered=entered,
        label="r2-dead",
    )
    results: list[EventDeliveryResult] = []
    waiter_done = threading.Event()

    def _important() -> None:
        results.append(
            sink.publish(_event("r2-dead-wait"), priority=EventPriority.IMPORTANT),
        )
        waiter_done.set()

    thread = threading.Thread(target=_important, name="r2-dead-wait")
    thread.start()
    thread.join(timeout=0.25)
    assert not waiter_done.is_set()
    buffer.force_worker_failure.set()
    release.set()
    assert waiter_done.wait(timeout=_SYNC_TIMEOUT)
    assert results
    assert results[0].disposition is EventDeliveryDisposition.REJECTED
    assert sink.health_state() is EventSinkHealthState.UNHEALTHY


def test_r2_no_post_close_user_delivery_ids() -> None:
    downstream = InMemoryEventSink()
    policy = EventDeliveryPolicy(max_capacity=4, critical_reserved_capacity=0)
    sink = BoundedEventSink(downstream, policy)
    pre_ids = {f"pre-{index}" for index in range(3)}
    for label in sorted(pre_ids):
        assert (
            sink.publish(_event(label), priority=EventPriority.BEST_EFFORT).disposition
            is EventDeliveryDisposition.ACCEPTED
        )
    sink.close()
    assert (
        sink.publish(
            _event("must-not-deliver"), priority=EventPriority.BEST_EFFORT
        ).disposition
        is EventDeliveryDisposition.REJECTED
    )
    delivered = {event.event_id for _, event in downstream.records}
    assert "must-not-deliver" not in delivered
    assert pre_ids.issubset(delivered)


class _BlockingEventDeliveryBuffer(Generic[_TBuffered]):
    """Conforming fault buffer: holds physical enqueue until released."""

    def __init__(
        self,
        *,
        capacity: int,
        should_release: Callable[[], bool] | None = None,
    ) -> None:
        self._inner: QueueBackedEventDeliveryBuffer[_TBuffered] = (
            QueueBackedEventDeliveryBuffer(capacity=capacity)
        )
        self.hold = threading.Event()
        self.enqueue_entered = threading.Event()
        self.recorded_operations: list[str] = []
        self._order_lock = threading.Lock()
        self._hold_condition = threading.Condition()
        self._should_release = should_release

    @property
    def capacity(self) -> int:
        return self._inner.capacity

    @property
    def pending_depth(self) -> int:
        return self._inner.pending_depth

    def release(self) -> None:
        self.hold.clear()
        with self._hold_condition:
            self._hold_condition.notify_all()

    def _gate(self) -> None:
        if not self.hold.is_set():
            return
        self.enqueue_entered.set()
        with self._hold_condition:
            while self.hold.is_set():
                if self._should_release is not None and self._should_release():
                    self.hold.clear()
                    self._hold_condition.notify_all()
                    break
                self._hold_condition.wait(timeout=0.05)

    def _record(self, operation: str) -> None:
        with self._order_lock:
            self.recorded_operations.append(operation)

    def enqueue_item_nowait(self, item: _TBuffered) -> None:
        self._gate()
        self._record("enqueue_item")
        self._inner.enqueue_item_nowait(item)

    def enqueue_item(self, item: _TBuffered, *, timeout: float) -> None:
        self._gate()
        self._record("enqueue_item")
        self._inner.enqueue_item(item, timeout=timeout)

    def enqueue_shutdown_nowait(self) -> None:
        self._gate()
        self._record("enqueue_shutdown")
        self._inner.enqueue_shutdown_nowait()

    def take_next(self) -> EventDeliveryBufferEntry[_TBuffered]:
        return self._inner.take_next()

    def acknowledge_processed(self) -> None:
        self._inner.acknowledge_processed()


class _WorkerDeathEventDeliveryBuffer(Generic[_TBuffered]):
    """Conforming buffer that fails the drain worker after acknowledge."""

    def __init__(self, *, capacity: int) -> None:
        self._inner: QueueBackedEventDeliveryBuffer[_TBuffered] = (
            QueueBackedEventDeliveryBuffer(capacity=capacity)
        )
        self.force_worker_failure = threading.Event()

    @property
    def capacity(self) -> int:
        return self._inner.capacity

    @property
    def pending_depth(self) -> int:
        return self._inner.pending_depth

    def enqueue_item_nowait(self, item: _TBuffered) -> None:
        self._inner.enqueue_item_nowait(item)

    def enqueue_item(self, item: _TBuffered, *, timeout: float) -> None:
        self._inner.enqueue_item(item, timeout=timeout)

    def enqueue_shutdown_nowait(self) -> None:
        self._inner.enqueue_shutdown_nowait()

    def take_next(self) -> EventDeliveryBufferEntry[_TBuffered]:
        return self._inner.take_next()

    def acknowledge_processed(self) -> None:
        self._inner.acknowledge_processed()
        if self.force_worker_failure.is_set():
            raise RuntimeError("simulated event delivery drain worker failure")


class _CountingEventDeliveryBuffer(Generic[_TBuffered]):
    """Distinct conforming buffer used only to prove injection without subclassing."""

    def __init__(self, *, capacity: int) -> None:
        self._inner: QueueBackedEventDeliveryBuffer[_TBuffered] = (
            QueueBackedEventDeliveryBuffer(capacity=capacity)
        )
        self.enqueue_item_calls = 0

    @property
    def capacity(self) -> int:
        return self._inner.capacity

    @property
    def pending_depth(self) -> int:
        return self._inner.pending_depth

    def enqueue_item_nowait(self, item: _TBuffered) -> None:
        self.enqueue_item_calls += 1
        self._inner.enqueue_item_nowait(item)

    def enqueue_item(self, item: _TBuffered, *, timeout: float) -> None:
        self.enqueue_item_calls += 1
        self._inner.enqueue_item(item, timeout=timeout)

    def enqueue_shutdown_nowait(self) -> None:
        self._inner.enqueue_shutdown_nowait()

    def take_next(self) -> EventDeliveryBufferEntry[_TBuffered]:
        return self._inner.take_next()

    def acknowledge_processed(self) -> None:
        self._inner.acknowledge_processed()


def test_r3_pending_enqueue_shutdown_timeout_fails_closed() -> None:
    policy = EventDeliveryPolicy(
        max_capacity=4,
        critical_reserved_capacity=1,
        critical_completion_timeout_seconds=30.0,
        drain_shutdown_timeout_seconds=0.2,
    )
    downstream_closed = threading.Event()

    class _Downstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.COMPLETION,
            )

        def close(self) -> None:
            downstream_closed.set()

    buffer = _BlockingEventDeliveryBuffer(capacity=policy.max_capacity)
    sink = BoundedEventSink(_Downstream(), policy, buffer=buffer)
    buffer.hold.set()
    publish_done = threading.Event()

    def _producer() -> None:
        sink.publish(_event("r3-timeout"), priority=EventPriority.BEST_EFFORT)
        publish_done.set()

    thread = threading.Thread(target=_producer, name="r3-producer")
    thread.start()
    assert buffer.enqueue_entered.wait(timeout=_SYNC_TIMEOUT)

    with pytest.raises(EventDeliveryBoundaryError) as exc_info:
        sink.close()
    assert "pending physical delivery enqueue" in str(exc_info.value)
    assert sink.health_state() is EventSinkHealthState.UNHEALTHY
    assert "enqueue_shutdown" not in buffer.recorded_operations
    assert not downstream_closed.is_set()

    with pytest.raises(EventDeliveryBoundaryError):
        sink.close()
    post = sink.publish(_event("r3-post-fail"), priority=EventPriority.BEST_EFFORT)
    assert post.disposition is EventDeliveryDisposition.REJECTED

    buffer.release()
    assert publish_done.wait(timeout=_SYNC_TIMEOUT)
    assert "enqueue_shutdown" not in buffer.recorded_operations
    thread.join(timeout=_SYNC_TIMEOUT)


def test_r3_pending_enqueue_drains_before_deadline_shutdown_succeeds() -> None:
    policy = EventDeliveryPolicy(
        max_capacity=4,
        critical_reserved_capacity=1,
        critical_completion_timeout_seconds=30.0,
        drain_shutdown_timeout_seconds=2.0,
    )

    class _SinkClosedProbe:
        def __init__(self) -> None:
            self.sink: BoundedEventSink | None = None

        def __call__(self) -> bool:
            return self.sink is not None and self.sink.closed

    closed_probe = _SinkClosedProbe()
    buffer = _BlockingEventDeliveryBuffer(
        capacity=policy.max_capacity,
        should_release=closed_probe,
    )
    sink = BoundedEventSink(InMemoryEventSink(), policy, buffer=buffer)
    closed_probe.sink = sink
    buffer.hold.set()
    admitted = threading.Event()

    def _producer() -> None:
        result = sink.publish(_event("r3-drain"), priority=EventPriority.BEST_EFFORT)
        assert result.disposition is EventDeliveryDisposition.ACCEPTED
        admitted.set()

    thread = threading.Thread(target=_producer, name="r3-drain-producer")
    thread.start()
    assert buffer.enqueue_entered.wait(timeout=_SYNC_TIMEOUT)
    sink.close()
    assert admitted.wait(timeout=_SYNC_TIMEOUT)
    assert sink.health_state() is EventSinkHealthState.HEALTHY
    assert "enqueue_item" in buffer.recorded_operations
    assert "enqueue_shutdown" in buffer.recorded_operations
    assert buffer.recorded_operations.index(
        "enqueue_item"
    ) < buffer.recorded_operations.index("enqueue_shutdown")
    thread.join(timeout=_SYNC_TIMEOUT)


def test_custom_buffer_pluginability_preserves_qos_admission() -> None:
    policy = EventDeliveryPolicy(max_capacity=4, critical_reserved_capacity=1)
    custom = _CountingEventDeliveryBuffer(capacity=policy.max_capacity)
    sink = BoundedEventSink(InMemoryEventSink(), policy, buffer=custom)
    accepted = sink.publish(_event("plugin-be"), priority=EventPriority.BEST_EFFORT)
    assert accepted.disposition is EventDeliveryDisposition.ACCEPTED
    assert custom.enqueue_item_calls == 1
    sink.close()
    assert custom.enqueue_item_calls == 1


def test_architecture_gate_r3_no_silent_pending_drain_timeout() -> None:
    src = (
        Path(__file__).resolve().parents[4]
        / "intergrax"
        / "runtime"
        / "observability"
        / "event_delivery"
        / "bounded_event_sink.py"
    ).read_text(encoding="utf-8")
    assert "pending physical delivery enqueue did not drain" in src
    drain_fn = src.split("def _wait_pending_physical_enqueue_drain", 1)[1].split(
        "\n    def ", 1
    )[0]
    assert "if remaining <= 0:" in drain_fn
    assert (
        "return\n"
        not in drain_fn.split("if remaining <= 0:", 1)[1].split(
            "self._quota_condition.wait", 1
        )[0]
    )
