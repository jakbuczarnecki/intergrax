# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded drain terminal: ``EventSinkPort`` → ``EventExportSinkPort``."""

from __future__ import annotations

import time

from intergrax.contracts.event_delivery import (
    DeliverableEvent,
    EventDeliveryDisposition,
    EventDeliveryResult,
    EventExportSinkPort,
    EventPriority,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.observability.event_delivery.async_export_runner import (
    AsyncExportRunner,
)
from intergrax.runtime.observability.event_delivery.delivery_metrics import (
    InternalDeliveryMetrics,
)


class RuntimeEventExportSink:
    """
    Process-local bridge from bounded backpressure to pluggable export sinks.

    Export failures never propagate to the execution plane; they are recorded on
    ``InternalDeliveryMetrics`` only.
    """

    def __init__(
        self,
        export_sink: EventExportSinkPort,
        *,
        delivery_metrics: InternalDeliveryMetrics | None = None,
    ) -> None:
        self._export_sink = export_sink
        self._delivery_metrics = delivery_metrics
        self._runner = AsyncExportRunner()
        self._closed = False
        self._pending_exports = 0

    @property
    def export_sink(self) -> EventExportSinkPort:
        return self._export_sink

    @property
    def closed(self) -> bool:
        return self._closed

    def publish(
        self,
        event: DeliverableEvent,
        *,
        priority: EventPriority,
        deadline: float | None = None,
    ) -> EventDeliveryResult:
        """Legacy ``EventSinkPort`` entry — use ``deliver_bounded`` from the drain worker."""
        return EventDeliveryResult(
            disposition=EventDeliveryDisposition.REJECTED,
            priority=priority,
            buffered_depth=0,
        )

    def deliver_bounded(
        self,
        source_event: RuntimeEvent,
        deliverable: DeliverableEvent,
        *,
        priority: EventPriority,
    ) -> EventDeliveryResult:
        if self._closed:
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.REJECTED,
                priority=priority,
                buffered_depth=0,
            )
        self._pending_exports += 1
        depth = self._pending_exports
        started = time.monotonic()
        metrics = self._delivery_metrics
        if metrics is not None:
            metrics.record_export_attempt()
        try:
            self._runner.run(self._export_sink.export(source_event))
            latency = time.monotonic() - started
            metrics = self._delivery_metrics
            if metrics is not None:
                metrics.record_export_accepted(
                    latency_seconds=latency,
                    queue_depth=depth,
                )
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=depth,
            )
        except Exception:
            latency = time.monotonic() - started
            metrics = self._delivery_metrics
            if metrics is not None:
                dropped = priority is EventPriority.BEST_EFFORT
                metrics.record_export_failed(
                    latency_seconds=latency,
                    queue_depth=depth,
                    dropped=dropped,
                )
            disposition = (
                EventDeliveryDisposition.DROPPED
                if priority is EventPriority.BEST_EFFORT
                else EventDeliveryDisposition.REJECTED
            )
            return EventDeliveryResult(
                disposition=disposition,
                priority=priority,
                buffered_depth=depth,
            )
        finally:
            self._pending_exports = max(0, self._pending_exports - 1)

    def flush_sync(self) -> None:
        if self._closed:
            return
        started = time.monotonic()
        metrics = self._delivery_metrics
        try:
            self._runner.run(self._export_sink.flush())
        except Exception:
            if metrics is not None:
                metrics.record_export_flush_failed()
        finally:
            elapsed = time.monotonic() - started
            if metrics is not None:
                metrics.record_export_flush_duration(elapsed)

    def close(self) -> None:
        if self._closed:
            return
        self.flush_sync()
        try:
            self._runner.run(self._export_sink.close())
        finally:
            self._runner.shutdown()
            self._closed = True
