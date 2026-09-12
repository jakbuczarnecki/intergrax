# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-process sink for qualification and wiring tests."""

from __future__ import annotations

import threading
import time

from intergrax.contracts.event_delivery import (
    DeliverableEvent,
    EventDeliveryDisposition,
    EventDeliveryResult,
    EventPriority,
    EventSinkPort,
)


class InMemoryEventSink:
    """Synchronous collector implementing ``EventSinkPort``."""

    def __init__(self, *, consume_delay_seconds: float = 0.0) -> None:
        self._consume_delay_seconds = consume_delay_seconds
        self._lock = threading.Lock()
        self._records: list[tuple[EventPriority, DeliverableEvent]] = []
        self._closed = False

    @property
    def records(self) -> list[tuple[EventPriority, DeliverableEvent]]:
        with self._lock:
            return list(self._records)

    def publish(
        self,
        event: DeliverableEvent,
        *,
        priority: EventPriority,
        deadline: float | None = None,
    ) -> EventDeliveryResult:
        if self._closed:
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.REJECTED,
                priority=priority,
                buffered_depth=0,
            )
        if self._consume_delay_seconds > 0:
            time.sleep(self._consume_delay_seconds)
        with self._lock:
            self._records.append((priority, event))
            depth = len(self._records)
        return EventDeliveryResult(
            disposition=EventDeliveryDisposition.ACCEPTED,
            priority=priority,
            buffered_depth=depth,
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True


def assert_event_sink_port(sink: EventSinkPort) -> EventSinkPort:
    return sink
