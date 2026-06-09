# © Artur Czarnecki. All rights reserved.

"""Capacity signal collector (ECP-2.*)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from intergrax.runtime.capacity.contracts import CapacitySignal, ScalingTarget
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType


PublishFn = Callable[[RuntimeEvent], None]


class CapacitySignalCollector:
    """Aggregate runtime signals for scaling evaluation."""

    def __init__(
        self,
        *,
        publish: PublishFn | None = None,
        queue_depth_provider: Callable[[], float] | None = None,
    ) -> None:
        self._publish = publish
        self._queue_depth_provider = queue_depth_provider
        self._backpressure_count = 0

    def record_backpressure(self) -> None:
        self._backpressure_count += 1

    def collect(
        self,
        *,
        backpressure_rate: float | None = None,
    ) -> list[CapacitySignal]:
        signals: list[CapacitySignal] = []
        rate = backpressure_rate if backpressure_rate is not None else float(self._backpressure_count)
        signals.append(
            CapacitySignal(
                target=ScalingTarget.ORCHESTRATION_CEILING,
                metric_name="graph_backpressure_rate",
                value=rate,
            )
        )
        if self._queue_depth_provider is not None:
            depth = self._queue_depth_provider()
            signals.append(
                CapacitySignal(
                    target=ScalingTarget.CELERY_POOL,
                    metric_name="queue_depth",
                    value=depth,
                )
            )
        if self._publish is not None:
            for signal in signals:
                self._publish(
                    RuntimeEvent(
                        event_type=RuntimeEventType.TASK_PROGRESS,
                        tenant_id="harness",
                        task_id=signal.signal_id,
                        run_id=signal.signal_id,
                        phase=ExecutionPhase.EXECUTION,
                        payload={
                            "event_kind": "CAPACITY_SIGNAL_COLLECTED",
                            "target": signal.target.value,
                            "metric_name": signal.metric_name,
                            "value": signal.value,
                        },
                    )
                )
        return signals
