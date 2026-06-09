# © Artur Czarnecki. All rights reserved.

"""Capacity signal collector (ECP-2.*)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from intergrax.runtime.capacity.contracts import CapacitySignal, ScalingTarget
from intergrax.runtime.capacity.events import PublishFn, publish_capacity_signal_collected


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
                publish_capacity_signal_collected(self._publish, signal)
        return signals
