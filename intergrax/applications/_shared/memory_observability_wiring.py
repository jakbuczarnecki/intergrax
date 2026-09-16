# © Artur Czarnecki. All rights reserved.

"""Shared memory observability composition (MEM-ENT-12)."""

from __future__ import annotations

from intergrax.memory.contracts.memory_observability import (
    MemoryObservabilitySink,
    NoOpMemoryObservabilitySink,
)
from intergrax.memory.memory_diagnostic_emitter import (
    MemoryDiagnosticEmitter,
    default_memory_diagnostic_emitter,
)

__all__ = [
    "resolve_memory_diagnostic_emitter",
    "resolve_memory_observability_sink",
]


def resolve_memory_observability_sink(
    sink: MemoryObservabilitySink | None = None,
) -> MemoryObservabilitySink:
    return sink if sink is not None else NoOpMemoryObservabilitySink()


def resolve_memory_diagnostic_emitter(
    *,
    sink: MemoryObservabilitySink | None = None,
    emitter: MemoryDiagnosticEmitter | None = None,
) -> MemoryDiagnosticEmitter:
    if emitter is not None:
        return emitter
    return default_memory_diagnostic_emitter(resolve_memory_observability_sink(sink))
