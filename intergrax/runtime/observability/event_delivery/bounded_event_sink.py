# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Process-local bounded buffer between producers and a downstream ``EventSinkPort``."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Literal

from intergrax.contracts.event_delivery import (
    DeliverableEvent,
    EventDeliveryAdmissionPolicyPort,
    EventDeliveryBoundaryError,
    EventDeliveryBoundaryFailureKind,
    EventDeliveryDisposition,
    EventDeliveryLateFailure,
    EventDeliveryLateFailureStage,
    EventDeliveryObligation,
    EventDeliveryObligationPolicyPort,
    EventDeliveryPolicy,
    EventDeliveryPostAdmissionFailureObserverPort,
    EventDeliveryResult,
    EventPriority,
    EventSinkHealthPort,
    EventSinkHealthState,
    EventSinkPort,
    effective_event_delivery_obligation,
)
from intergrax.runtime.observability.event_delivery.enterprise_default_event_delivery_admission_policy import (
    EnterpriseDefaultEventDeliveryAdmissionPolicy,
)
from intergrax.runtime.observability.event_delivery.enterprise_default_event_delivery_obligation_policy import (
    EnterpriseDefaultEventDeliveryObligationPolicy,
)
from intergrax.runtime.observability.event_delivery.event_sink_health import (
    MutableEventSinkHealth,
)
from intergrax.runtime.observability.event_delivery.logging_post_admission_failure_observer import (
    LoggingEventDeliveryPostAdmissionFailureObserver,
)

logger = logging.getLogger(__name__)


class _CompletionSignal:
    __slots__ = ("_event", "_result", "_boundary_error", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._result: EventDeliveryResult | None = None
        self._boundary_error: EventDeliveryBoundaryError | None = None

    def deliver_result(self, result: EventDeliveryResult) -> None:
        with self._lock:
            self._result = result
        self._event.set()

    def deliver_boundary(self, error: EventDeliveryBoundaryError) -> None:
        with self._lock:
            self._boundary_error = error
        self._event.set()

    def wait(self, deadline: float) -> Literal["timeout", "result", "boundary"]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "timeout"
        if not self._event.wait(timeout=remaining):
            return "timeout"
        with self._lock:
            if self._boundary_error is not None:
                return "boundary"
            if self._result is not None:
                return "result"
        return "timeout"

    @property
    def result(self) -> EventDeliveryResult | None:
        with self._lock:
            return self._result

    @property
    def boundary_error(self) -> EventDeliveryBoundaryError | None:
        with self._lock:
            return self._boundary_error


@dataclass(frozen=True, slots=True)
class _QueuedItem:
    priority: EventPriority
    event: DeliverableEvent
    obligation: EventDeliveryObligation
    completion_deadline: float | None
    completion: _CompletionSignal | None


class BoundedEventSink:
    """
    Bounded queue + background drainer.

    Producers never hold execution permits while waiting on a slow downstream sink
    beyond the policy-bound wait for IMPORTANT events (ADMISSION) or completion
    deadline for CRITICAL (COMPLETION).
    """

    def __init__(
        self,
        downstream: EventSinkPort,
        policy: EventDeliveryPolicy,
        *,
        obligation_policy: EventDeliveryObligationPolicyPort | None = None,
        admission_policy: EventDeliveryAdmissionPolicyPort | None = None,
        late_failure_observer: EventDeliveryPostAdmissionFailureObserverPort
        | None = None,
        health: EventSinkHealthPort | None = None,
    ) -> None:
        self._downstream = downstream
        self._policy = policy
        self._obligation_policy = (
            obligation_policy
            if obligation_policy is not None
            else EnterpriseDefaultEventDeliveryObligationPolicy()
        )
        self._admission_policy = (
            admission_policy
            if admission_policy is not None
            else EnterpriseDefaultEventDeliveryAdmissionPolicy()
        )
        self._late_failure_observer = (
            late_failure_observer
            if late_failure_observer is not None
            else LoggingEventDeliveryPostAdmissionFailureObserver()
        )
        self._health = health if health is not None else MutableEventSinkHealth()
        self._queue: queue.Queue[_QueuedItem | None] = queue.Queue(
            maxsize=policy.max_capacity,
        )
        self._non_critical_buffered = 0
        self._buffered_count_lock = threading.Lock()
        self._stop = threading.Event()
        self._worker_drained_normally = threading.Event()
        self._worker = threading.Thread(
            target=self._drain_loop, name="w5a-event-drain", daemon=True
        )
        self._worker.start()

    @property
    def pending_depth(self) -> int:
        return self._queue.qsize()

    @property
    def closed(self) -> bool:
        return self._stop.is_set()

    def health_state(self) -> EventSinkHealthState:
        return self._health.health_state()

    def publish(
        self,
        event: DeliverableEvent,
        *,
        priority: EventPriority,
        deadline: float | None = None,
    ) -> EventDeliveryResult:
        obligation = effective_event_delivery_obligation(
            priority, self._obligation_policy
        )
        depth = self.pending_depth

        if self._health.health_state() is EventSinkHealthState.UNHEALTHY:
            return self._make_result(
                disposition=EventDeliveryDisposition.REJECTED,
                priority=priority,
                obligation=obligation,
                buffered_depth=depth,
            )

        if not self._stop.is_set() and not self._worker.is_alive():
            self._health.mark_unhealthy()
            return self._make_result(
                disposition=EventDeliveryDisposition.REJECTED,
                priority=priority,
                obligation=obligation,
                buffered_depth=depth,
            )

        if self._stop.is_set():
            return self._make_result(
                disposition=EventDeliveryDisposition.REJECTED,
                priority=priority,
                obligation=obligation,
                buffered_depth=depth,
            )

        completion: _CompletionSignal | None = None
        completion_deadline = deadline
        if obligation is EventDeliveryObligation.COMPLETION:
            completion = _CompletionSignal()
            if completion_deadline is None:
                completion_deadline = (
                    time.monotonic() + self._policy.critical_completion_timeout_seconds
                )

        item = _QueuedItem(
            priority=priority,
            event=event,
            obligation=obligation,
            completion_deadline=completion_deadline,
            completion=completion,
        )

        if priority is EventPriority.BEST_EFFORT:
            if not self._try_admit_non_critical_nowait(item):
                return self._make_result(
                    disposition=EventDeliveryDisposition.DROPPED,
                    priority=priority,
                    obligation=obligation,
                    buffered_depth=depth,
                )
            return self._make_result(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                obligation=obligation,
                buffered_depth=self.pending_depth,
            )

        if priority is EventPriority.CRITICAL:
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                return self._make_result(
                    disposition=EventDeliveryDisposition.REJECTED,
                    priority=priority,
                    obligation=obligation,
                    buffered_depth=depth,
                )
            assert completion is not None and completion_deadline is not None
            return self._wait_for_completion(
                completion=completion,
                deadline=completion_deadline,
                priority=priority,
                obligation=obligation,
                event=event,
            )

        if self._non_critical_at_capacity():
            return self._make_result(
                disposition=EventDeliveryDisposition.DEFERRED,
                priority=priority,
                obligation=obligation,
                buffered_depth=depth,
            )
        timeout = self._important_timeout(deadline)
        try:
            self._queue.put(item, timeout=timeout)
            with self._buffered_count_lock:
                self._non_critical_buffered += 1
        except queue.Full:
            return self._make_result(
                disposition=EventDeliveryDisposition.DEFERRED,
                priority=priority,
                obligation=obligation,
                buffered_depth=depth,
            )
        if obligation is EventDeliveryObligation.COMPLETION:
            assert completion is not None and completion_deadline is not None
            return self._wait_for_completion(
                completion=completion,
                deadline=completion_deadline,
                priority=priority,
                obligation=obligation,
                event=event,
            )
        return self._make_result(
            disposition=EventDeliveryDisposition.ACCEPTED,
            priority=priority,
            obligation=obligation,
            buffered_depth=self.pending_depth,
        )

    def close(self) -> None:
        if self._stop.is_set():
            self._raise_if_shutdown_not_successful()
            return
        if not self._worker.is_alive() and not self._worker_drained_normally.is_set():
            self._stop.set()
            self._health.mark_unhealthy()
            raise EventDeliveryBoundaryError(
                kind=EventDeliveryBoundaryFailureKind.SINK_UNAVAILABLE,
                message="bounded event drain worker is not alive",
            )
        self._stop.set()
        self._enqueue_shutdown_sentinel()
        self._worker.join(timeout=self._policy.drain_shutdown_timeout_seconds)
        self._validate_worker_shutdown_after_join()
        try:
            self._downstream.close()
        except EventDeliveryBoundaryError:
            raise
        except Exception as exc:
            raise EventDeliveryBoundaryError(
                kind=EventDeliveryBoundaryFailureKind.INTERNAL_ERROR,
                message="downstream close failed",
            ) from exc

    def _wait_for_completion(
        self,
        *,
        completion: _CompletionSignal,
        deadline: float,
        priority: EventPriority,
        obligation: EventDeliveryObligation,
        event: DeliverableEvent,
    ) -> EventDeliveryResult:
        outcome = completion.wait(deadline)
        if outcome == "boundary":
            error = completion.boundary_error
            assert error is not None
            raise error
        if outcome == "result":
            result = completion.result
            assert result is not None
            return result
        return self._make_result(
            disposition=EventDeliveryDisposition.REJECTED,
            priority=priority,
            obligation=obligation,
            buffered_depth=self.pending_depth,
        )

    def _enqueue_shutdown_sentinel(self) -> None:
        """Ensure the drainer observes shutdown even when the buffer is saturated."""
        deadline = time.monotonic() + self._policy.drain_shutdown_timeout_seconds
        while self._worker.is_alive():
            if time.monotonic() >= deadline:
                self._health.mark_unhealthy()
                raise EventDeliveryBoundaryError(
                    kind=EventDeliveryBoundaryFailureKind.INTERNAL_ERROR,
                    message="shutdown sentinel enqueue deadline expired",
                )
            try:
                self._queue.put_nowait(None)
                return
            except queue.Full:
                time.sleep(0.001)
        self._health.mark_unhealthy()
        raise EventDeliveryBoundaryError(
            kind=EventDeliveryBoundaryFailureKind.SINK_UNAVAILABLE,
            message="bounded event drain worker died before shutdown sentinel was enqueued",
        )

    def _validate_worker_shutdown_after_join(self) -> None:
        if self._worker.is_alive():
            self._health.mark_unhealthy()
            raise EventDeliveryBoundaryError(
                kind=EventDeliveryBoundaryFailureKind.INTERNAL_ERROR,
                message="bounded event drain worker did not terminate within drain deadline",
            )
        if not self._worker_drained_normally.is_set():
            self._health.mark_unhealthy()
            raise EventDeliveryBoundaryError(
                kind=EventDeliveryBoundaryFailureKind.SINK_UNAVAILABLE,
                message="bounded event drain worker did not complete normal shutdown drain",
            )
        if self._health.health_state() is EventSinkHealthState.UNHEALTHY:
            raise EventDeliveryBoundaryError(
                kind=EventDeliveryBoundaryFailureKind.SINK_UNAVAILABLE,
                message="bounded event drain worker shutdown completed with unhealthy sink",
            )

    def _raise_if_shutdown_not_successful(self) -> None:
        if (
            self._worker_drained_normally.is_set()
            and self._health.health_state() is EventSinkHealthState.HEALTHY
        ):
            return
        self._health.mark_unhealthy()
        raise EventDeliveryBoundaryError(
            kind=EventDeliveryBoundaryFailureKind.SINK_UNAVAILABLE,
            message="bounded event sink shutdown did not complete successfully",
        )

    def _max_non_critical_buffered(self) -> int:
        return self._admission_policy.max_non_critical_buffered_events(self._policy)

    def _non_critical_at_capacity(self) -> bool:
        with self._buffered_count_lock:
            return self._non_critical_buffered >= self._max_non_critical_buffered()

    def _try_admit_non_critical_nowait(self, item: _QueuedItem) -> bool:
        with self._buffered_count_lock:
            if self._non_critical_buffered >= self._max_non_critical_buffered():
                return False
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                return False
            self._non_critical_buffered += 1
            return True

    def _dequeue_accounting(self, item: _QueuedItem | None) -> None:
        if item is None:
            return
        if item.priority is not EventPriority.CRITICAL:
            with self._buffered_count_lock:
                self._non_critical_buffered -= 1

    def _important_timeout(self, deadline: float | None) -> float:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 0.0
            return min(remaining, self._policy.important_wait_timeout_seconds)
        return self._policy.important_wait_timeout_seconds

    def _make_result(
        self,
        *,
        disposition: EventDeliveryDisposition,
        priority: EventPriority,
        obligation: EventDeliveryObligation,
        buffered_depth: int,
    ) -> EventDeliveryResult:
        return EventDeliveryResult(
            disposition=disposition,
            priority=priority,
            buffered_depth=buffered_depth,
            obligation=obligation,
        )

    def _normalize_downstream_result(
        self,
        result: EventDeliveryResult,
        *,
        priority: EventPriority,
        obligation: EventDeliveryObligation,
        buffered_depth: int,
    ) -> EventDeliveryResult:
        if result.priority is priority and result.obligation is obligation:
            if result.buffered_depth == buffered_depth:
                return result
        return self._make_result(
            disposition=result.disposition,
            priority=priority,
            obligation=obligation,
            buffered_depth=buffered_depth,
        )

    def _notify_late_failure(
        self,
        *,
        deliverable: DeliverableEvent,
        priority: EventPriority,
        disposition: EventDeliveryDisposition,
        boundary_kind: EventDeliveryBoundaryFailureKind | None,
    ) -> None:
        failure = EventDeliveryLateFailure(
            deliverable=deliverable,
            priority=priority,
            disposition=disposition,
            stage=EventDeliveryLateFailureStage.DOWNSTREAM_PUBLISH,
            boundary_kind=boundary_kind,
        )
        try:
            self._late_failure_observer.on_late_failure(failure)
        except Exception:
            logger.exception("late admission failure observer failed")

    def _completion_deadline_expired(self, item: _QueuedItem) -> bool:
        if item.obligation is not EventDeliveryObligation.COMPLETION:
            return False
        if item.completion_deadline is None:
            return False
        return time.monotonic() >= item.completion_deadline

    def _process_item(self, item: _QueuedItem) -> None:
        if self._completion_deadline_expired(item):
            boundary_error = EventDeliveryBoundaryError(
                kind=EventDeliveryBoundaryFailureKind.COMPLETION_TIMEOUT,
                message="completion deadline expired before downstream work",
                deliverable_event_id=item.event.event_id,
            )
            if item.completion is not None:
                item.completion.deliver_boundary(boundary_error)
                return
            self._notify_late_failure(
                deliverable=item.event,
                priority=item.priority,
                disposition=EventDeliveryDisposition.REJECTED,
                boundary_kind=EventDeliveryBoundaryFailureKind.COMPLETION_TIMEOUT,
            )
            return

        downstream_deadline = item.completion_deadline
        try:
            raw = self._downstream.publish(
                item.event,
                priority=item.priority,
                deadline=downstream_deadline,
            )
        except EventDeliveryBoundaryError as boundary_error:
            if item.completion is not None:
                item.completion.deliver_boundary(boundary_error)
                return
            self._notify_late_failure(
                deliverable=item.event,
                priority=item.priority,
                disposition=EventDeliveryDisposition.REJECTED,
                boundary_kind=boundary_error.kind,
            )
            return
        except Exception as exc:
            logger.exception("unexpected downstream delivery defect")
            try:
                raise EventDeliveryBoundaryError(
                    kind=EventDeliveryBoundaryFailureKind.INTERNAL_ERROR,
                    message="unexpected downstream delivery defect",
                    deliverable_event_id=item.event.event_id,
                ) from exc
            except EventDeliveryBoundaryError as boundary_error:
                if item.completion is not None:
                    item.completion.deliver_boundary(boundary_error)
                    return
                self._notify_late_failure(
                    deliverable=item.event,
                    priority=item.priority,
                    disposition=EventDeliveryDisposition.REJECTED,
                    boundary_kind=EventDeliveryBoundaryFailureKind.INTERNAL_ERROR,
                )
                return

        result = self._normalize_downstream_result(
            raw,
            priority=item.priority,
            obligation=item.obligation,
            buffered_depth=self.pending_depth,
        )
        if item.completion is not None:
            item.completion.deliver_result(result)
            return
        if result.disposition is not EventDeliveryDisposition.ACCEPTED:
            self._notify_late_failure(
                deliverable=item.event,
                priority=item.priority,
                disposition=result.disposition,
                boundary_kind=None,
            )

    def _drain_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    self._dequeue_accounting(item)
                    if item is None:
                        self._worker_drained_normally.set()
                        break
                    self._process_item(item)
                finally:
                    self._queue.task_done()
        except Exception:
            self._health.mark_unhealthy()
            logger.exception("bounded event drain worker terminated unexpectedly")
