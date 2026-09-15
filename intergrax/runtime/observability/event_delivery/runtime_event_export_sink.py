# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded drain terminal: ``EventSinkPort`` → ``EventExportSinkPort``."""

from __future__ import annotations

import concurrent.futures
import time

from intergrax.contracts.event_delivery import (
    DeliverableEvent,
    EventDeliveryBoundaryError,
    EventDeliveryBoundaryFailureKind,
    EventDeliveryDisposition,
    EventDeliveryResult,
    EventExportSinkPort,
    EventPriority,
    effective_event_delivery_obligation,
)
from intergrax.runtime.observability.event_delivery.enterprise_default_event_delivery_obligation_policy import (
    EnterpriseDefaultEventDeliveryObligationPolicy,
)
from intergrax.runtime.observability.event_delivery.async_export_runner import (
    AsyncExportRunner,
)
from intergrax.runtime.observability.event_delivery.delivery_metrics import (
    InternalDeliveryMetrics,
)


_DEFAULT_OBLIGATION_POLICY = EnterpriseDefaultEventDeliveryObligationPolicy()


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
        obligation = effective_event_delivery_obligation(priority, _DEFAULT_OBLIGATION_POLICY)
        if self._closed:
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.REJECTED,
                priority=priority,
                buffered_depth=0,
                obligation=obligation,
            )
        if deadline is not None and deadline <= time.monotonic():
            raise EventDeliveryBoundaryError(
                kind=EventDeliveryBoundaryFailureKind.COMPLETION_TIMEOUT,
                message="export deadline already expired",
                deliverable_event_id=event.event_id,
            )

        self._pending_exports += 1
        depth = self._pending_exports
        started = time.monotonic()
        metrics = self._delivery_metrics
        if metrics is not None:
            metrics.record_export_attempt()
        export_timeout = 30.0
        if deadline is not None:
            export_timeout = deadline - time.monotonic()
        try:
            self._runner.run(
                self._export_sink.export(event.export_payload),
                timeout_seconds=export_timeout,
            )
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
                obligation=obligation,
            )
        except EventDeliveryBoundaryError:
            raise
        except concurrent.futures.TimeoutError as exc:
            raise EventDeliveryBoundaryError(
                kind=EventDeliveryBoundaryFailureKind.COMPLETION_TIMEOUT,
                message="export timed out",
                deliverable_event_id=event.event_id,
            ) from exc
        except Exception as exc:
            latency = time.monotonic() - started
            metrics = self._delivery_metrics
            if metrics is not None:
                dropped = priority is EventPriority.BEST_EFFORT
                metrics.record_export_failed(
                    latency_seconds=latency,
                    queue_depth=depth,
                    dropped=dropped,
                )
            raise EventDeliveryBoundaryError(
                kind=EventDeliveryBoundaryFailureKind.INTERNAL_ERROR,
                message="unexpected export sink defect",
                deliverable_event_id=event.event_id,
            ) from exc
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
