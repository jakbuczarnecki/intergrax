"""Reusable scoped timing profiler for pipeline stages."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.performance.contracts import (
    PipelinePhase,
)


class PipelineProfilerPort(Protocol):
    def measure(self, phase: PipelinePhase) -> Iterator[None]: ...

    def increment(self, counter: str, amount: int = 1) -> None: ...

    def seconds(self, phase: PipelinePhase) -> float: ...

    def counter(self, name: str) -> int: ...

    @property
    def enabled(self) -> bool: ...


@dataclass
class PipelineProfiler:
    """Accumulates exclusive timings per phase; supports nested measure() calls."""

    enabled: bool = True
    _timings: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _stack: list[tuple[str, float]] = field(default_factory=list)

    @contextmanager
    def measure(self, phase: PipelinePhase) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        phase_key = phase.value
        started = time.perf_counter()
        self._stack.append((phase_key, started))
        try:
            yield
        finally:
            if not self._stack:
                return
            active_phase, active_started = self._stack.pop()
            if active_phase != phase_key:
                return
            self._timings[phase_key] += time.perf_counter() - active_started

    def increment(self, counter: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        self._counters[counter] += amount

    def seconds(self, phase: PipelinePhase) -> float:
        return self._timings.get(phase.value, 0.0)

    def counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    def reset(self) -> None:
        self._timings.clear()
        self._counters.clear()
        self._stack.clear()


class NoOpPipelineProfiler:
    """Disabled profiler preserving call sites without measurement overhead."""

    enabled = False

    @contextmanager
    def measure(self, phase: PipelinePhase) -> Iterator[None]:
        yield

    def increment(self, counter: str, amount: int = 1) -> None:
        return None

    def seconds(self, phase: PipelinePhase) -> float:
        return 0.0

    def counter(self, name: str) -> int:
        return 0

    def reset(self) -> None:
        return None


def create_pipeline_profiler(enabled: bool) -> PipelineProfilerPort:
    if enabled:
        return PipelineProfiler(enabled=True)
    return NoOpPipelineProfiler()
