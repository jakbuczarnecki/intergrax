# © Artur Czarnecki. All rights reserved.

"""OBS-DELIVERY-QOS-SCALE — bounded priority delivery qualification."""

from __future__ import annotations

import threading
import time

import pytest

from intergrax.contracts.event_delivery import (
    DeliverableEvent,
    EventDeliveryAdmissionPolicyPort,
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

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SYNC_TIMEOUT = 10.0


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
    sink = BoundedEventSink(downstream, policy)
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
    with sink._quota_condition:  # noqa: SLF001 — deterministic lifecycle seam
        sink._worker = threading.Thread()  # noqa: SLF001 — not started → not alive
        sink._quota_condition.notify_all()
    assert waiter_done.wait(timeout=_SYNC_TIMEOUT)
    assert results
    assert results[0].disposition is EventDeliveryDisposition.REJECTED
    assert sink.health_state() is EventSinkHealthState.UNHEALTHY
    release.set()


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
