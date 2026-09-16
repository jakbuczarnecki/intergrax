# © Artur Czarnecki. All rights reserved.

"""Safe emission boundary for memory diagnostics (MEM-ENT-12)."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from intergrax.contracts.execution_identity import EventId, mint_event_id
from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticEvent,
    MemoryObservabilitySink,
    NoOpMemoryObservabilitySink,
)
from intergrax.utils.time_provider import SystemTimeProvider, TimeProvider

__all__ = [
    "MemoryDiagnosticEmitter",
    "MemoryOperationTimer",
    "default_memory_diagnostic_emitter",
]


@dataclass(slots=True)
class MemoryOperationTimer:
    _started_at: float = field(default_factory=perf_counter)

    def elapsed_seconds(self) -> float:
        return perf_counter() - self._started_at


@dataclass(slots=True)
class MemoryDiagnosticEmitter:
    """Isolates sink failures from authoritative memory operations."""

    _sink: MemoryObservabilitySink = field(default_factory=NoOpMemoryObservabilitySink)
    _time_provider: type[TimeProvider] = SystemTimeProvider

    @property
    def sink(self) -> MemoryObservabilitySink:
        return self._sink

    def emit(self, event: MemoryDiagnosticEvent) -> None:
        try:
            self._sink.record(event)
        except Exception:
            return None

    def new_event_id(self) -> EventId:
        return mint_event_id()

    def reference_time_iso(self) -> str:
        return self._time_provider.utc_now().isoformat()


def default_memory_diagnostic_emitter(
    sink: MemoryObservabilitySink | None = None,
) -> MemoryDiagnosticEmitter:
    resolved = sink if sink is not None else NoOpMemoryObservabilitySink()
    return MemoryDiagnosticEmitter(_sink=resolved)
