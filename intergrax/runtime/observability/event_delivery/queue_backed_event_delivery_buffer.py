# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default ``EventDeliveryBufferPort`` backed by a process-local bounded queue."""

from __future__ import annotations

import queue
from typing import Generic, TypeVar, cast

from intergrax.contracts.event_delivery import (
    EventDeliveryBufferCapacityExhausted,
    EventDeliveryBufferEntry,
)

TBuffered = TypeVar("TBuffered")

_SHUTDOWN = object()


class QueueBackedEventDeliveryBuffer(Generic[TBuffered]):
    """Queue-backed physical buffer for process-local event delivery.

    Storage/enqueue mechanics only — QoS, admission, health, and shutdown
    linearization remain owned by ``BoundedEventSink`` / ``EventDeliveryPolicy``.
    """

    def __init__(self, *, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be int")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._queue: queue.Queue[object] = queue.Queue(maxsize=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def pending_depth(self) -> int:
        return self._queue.qsize()

    def enqueue_item_nowait(self, item: TBuffered) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full as exc:
            raise EventDeliveryBufferCapacityExhausted from exc

    def enqueue_item(self, item: TBuffered, *, timeout: float) -> None:
        try:
            self._queue.put(item, timeout=timeout)
        except queue.Full as exc:
            raise EventDeliveryBufferCapacityExhausted from exc

    def enqueue_shutdown_nowait(self) -> None:
        try:
            self._queue.put_nowait(_SHUTDOWN)
        except queue.Full as exc:
            raise EventDeliveryBufferCapacityExhausted from exc

    def take_next(self) -> EventDeliveryBufferEntry[TBuffered]:
        raw = self._queue.get()
        if raw is _SHUTDOWN:
            return EventDeliveryBufferEntry(kind="shutdown", item=None)
        return EventDeliveryBufferEntry(kind="item", item=cast(TBuffered, raw))

    def acknowledge_processed(self) -> None:
        self._queue.task_done()
