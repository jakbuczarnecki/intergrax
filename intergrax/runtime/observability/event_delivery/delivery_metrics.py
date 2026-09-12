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
    export_accepted: int
    export_failed: int
    export_last_latency_seconds: float
    export_flush_duration_seconds: float
    export_queue_depth: int
    export_attempt_total: int
    export_success_total: int
    export_failed_total: int
    export_flush_total: int
    export_flush_failed_total: int
    export_latency_seconds: float
    exporter_kind: str


class InternalDeliveryMetrics:
    """Process-local delivery telemetry; not an EventSinkPort consumer."""

    def __init__(self, *, exporter_kind: str = "noop") -> None:
        self._lock = threading.Lock()
        self._exporter_kind = exporter_kind
        self._accepted = 0
        self._rejected = 0
        self._dropped = 0
        self._deferred = 0
        self._queue_depth = 0
        self._last_latency = 0.0
        self._export_accepted = 0
        self._export_failed = 0
        self._export_last_latency = 0.0
        self._export_flush_duration = 0.0
        self._export_queue_depth = 0
        self._export_attempt_total = 0
        self._export_success_total = 0
        self._export_failed_total = 0
        self._export_flush_total = 0
        self._export_flush_failed_total = 0

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

    def record_export_attempt(self) -> None:
        with self._lock:
            self._export_attempt_total += 1

    def record_export_accepted(
        self,
        *,
        latency_seconds: float,
        queue_depth: int,
    ) -> None:
        with self._lock:
            self._export_accepted += 1
            self._export_success_total += 1
            self._export_last_latency = latency_seconds
            self._export_queue_depth = queue_depth

    def record_export_failed(
        self,
        *,
        latency_seconds: float,
        queue_depth: int,
        dropped: bool,
    ) -> None:
        with self._lock:
            self._export_failed += 1
            self._export_failed_total += 1
            self._export_last_latency = latency_seconds
            self._export_queue_depth = queue_depth
            if dropped:
                self._dropped += 1

    def record_export_flush_duration(self, duration_seconds: float) -> None:
        with self._lock:
            self._export_flush_total += 1
            self._export_flush_duration = duration_seconds

    def record_export_flush_failed(self) -> None:
        with self._lock:
            self._export_flush_failed_total += 1

    def snapshot(self) -> DeliveryMetricsSnapshot:
        with self._lock:
            return DeliveryMetricsSnapshot(
                events_accepted=self._accepted,
                events_rejected=self._rejected,
                events_dropped=self._dropped,
                events_deferred=self._deferred,
                queue_depth=self._queue_depth,
                last_processing_latency_seconds=self._last_latency,
                export_accepted=self._export_accepted,
                export_failed=self._export_failed,
                export_last_latency_seconds=self._export_last_latency,
                export_flush_duration_seconds=self._export_flush_duration,
                export_queue_depth=self._export_queue_depth,
                export_attempt_total=self._export_attempt_total,
                export_success_total=self._export_success_total,
                export_failed_total=self._export_failed_total,
                export_flush_total=self._export_flush_total,
                export_flush_failed_total=self._export_flush_failed_total,
                export_latency_seconds=self._export_last_latency,
                exporter_kind=self._exporter_kind,
            )
