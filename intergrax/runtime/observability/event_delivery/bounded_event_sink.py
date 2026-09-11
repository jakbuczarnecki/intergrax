# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Process-local bounded buffer between producers and a downstream ``EventSinkPort``."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from intergrax.contracts.event_delivery import (
    CriticalEventDeliveryError,
    DeliverableEvent,
    EventDeliveryDisposition,
    EventDeliveryPolicy,
    EventDeliveryResult,
    EventPriority,
    EventSinkPort,
)


@dataclass(frozen=True, slots=True)
class _QueuedItem:
    priority: EventPriority
    event: DeliverableEvent
    enqueued_at: float


class BoundedEventSink:
    """
    Bounded queue + background drainer.

    Producers never hold execution permits while waiting on a slow downstream sink
    beyond the policy-bound wait for IMPORTANT events.
    """

    def __init__(
        self,
        downstream: EventSinkPort,
        policy: EventDeliveryPolicy,
    ) -> None:
        self._downstream = downstream
        self._policy = policy
        self._queue: queue.Queue[_QueuedItem | None] = queue.Queue(
            maxsize=policy.max_capacity,
        )
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._drain_loop, name="w5a-event-drain", daemon=True)
        self._worker.start()

    @property
    def pending_depth(self) -> int:
        return self._queue.qsize()

    def publish(
        self,
        event: DeliverableEvent,
        *,
        priority: EventPriority,
        deadline: float | None = None,
    ) -> EventDeliveryResult:
        if self._stop.is_set():
            if priority is EventPriority.CRITICAL:
                raise CriticalEventDeliveryError("sink closed with undelivered critical event")
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.REJECTED,
                priority=priority,
                buffered_depth=self.pending_depth,
            )

        item = _QueuedItem(priority=priority, event=event, enqueued_at=time.monotonic())

        if priority is EventPriority.BEST_EFFORT:
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                return EventDeliveryResult(
                    disposition=EventDeliveryDisposition.DROPPED,
                    priority=priority,
                    buffered_depth=self.pending_depth,
                )
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=self.pending_depth,
            )

        if priority is EventPriority.CRITICAL:
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                raise CriticalEventDeliveryError(
                    f"critical event {event.event_id} rejected: buffer full",
                )
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=self.pending_depth,
            )

        timeout = self._important_timeout(deadline)
        try:
            self._queue.put(item, timeout=timeout)
        except queue.Full:
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.DEFERRED,
                priority=priority,
                buffered_depth=self.pending_depth,
            )
        return EventDeliveryResult(
            disposition=EventDeliveryDisposition.ACCEPTED,
            priority=priority,
            buffered_depth=self.pending_depth,
        )

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._queue.put(None, timeout=max(self._policy.important_wait_timeout_seconds, 0.5))
        self._worker.join(timeout=10.0)
        if self._worker.is_alive():
            raise RuntimeError("bounded event drain worker did not terminate")

    def _important_timeout(self, deadline: float | None) -> float:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 0.0
            return min(remaining, self._policy.important_wait_timeout_seconds)
        return self._policy.important_wait_timeout_seconds

    def _drain_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    break
                self._downstream.publish(
                    item.event,
                    priority=item.priority,
                    deadline=None,
                )
            finally:
                self._queue.task_done()
