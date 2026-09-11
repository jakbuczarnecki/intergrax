# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Internal counters for observability delivery (never re-published on RuntimeEventBus)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from intergrax.contracts.event_delivery import EventDeliveryDisposition, EventDeliveryResult


@dataclass(frozen=True, slots=True)
class DeliveryMetricsSnapshot:
    events_accepted: int
    events_rejected: int
    events_dropped: int
    events_deferred: int
    queue_depth: int
    last_processing_latency_seconds: float


class InternalDeliveryMetrics:
    """Process-local delivery telemetry; not an EventSinkPort consumer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accepted = 0
        self._rejected = 0
        self._dropped = 0
        self._deferred = 0
        self._queue_depth = 0
        self._last_latency = 0.0

    def record(self, result: EventDeliveryResult, *, latency_seconds: float) -> None:
        with self._lock:
            if result.disposition is EventDeliveryDisposition.ACCEPTED:
                self._accepted += 1
            elif result.disposition is EventDeliveryDisposition.REJECTED:
                self._rejected += 1
            elif result.disposition is EventDeliveryDisposition.DROPPED:
                self._dropped += 1
            elif result.disposition is EventDeliveryDisposition.DEFERRED:
                self._deferred += 1
            self._queue_depth = result.buffered_depth
            self._last_latency = latency_seconds

    def snapshot(self) -> DeliveryMetricsSnapshot:
        with self._lock:
            return DeliveryMetricsSnapshot(
                events_accepted=self._accepted,
                events_rejected=self._rejected,
                events_dropped=self._dropped,
                events_deferred=self._deferred,
                queue_depth=self._queue_depth,
                last_processing_latency_seconds=self._last_latency,
            )
