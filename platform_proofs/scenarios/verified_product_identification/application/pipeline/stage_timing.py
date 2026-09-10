"""Monotonic stage timing helper."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter_ns
from typing import TypeVar

from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationStage,
    StageTimingObservedPayload,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.ports import (
    MonotonicClockPort,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.recorder import (
    ProductIdentificationObservationRecorder,
)

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class SystemMonotonicClock:
    def monotonic_ns(self) -> int:
        return perf_counter_ns()


def execute_timed_stage(
    *,
    recorder: ProductIdentificationObservationRecorder,
    clock: MonotonicClockPort,
    stage: ProductIdentificationStage,
    operation: Callable[[], _T],
) -> tuple[_T, int]:
    """Run ``operation`` callable, emit stage timing, return result and duration."""

    start = clock.monotonic_ns()
    result = operation()
    duration_ns = clock.monotonic_ns() - start
    if duration_ns < 0:
        duration_ns = 0
    recorder.record_payload(
        stage=stage,
        payload=StageTimingObservedPayload(stage=stage, duration_ns=duration_ns),
    )
    return result, duration_ns
