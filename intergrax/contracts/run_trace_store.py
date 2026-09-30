# © Artur Czarnecki. All rights reserved.

"""Neutral persisted run trace store ports (write/read)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.contracts.persisted_run_trace import (
    PersistedRun,
    RunMetadata,
    RunSummary,
)
from intergrax.contracts.tracing import TraceEvent


class RunTraceWriter(ABC):
    @abstractmethod
    def append_event(self, event: TraceEvent) -> None: ...

    @abstractmethod
    def finalize_run(self, run_id: str, metadata: RunMetadata) -> None: ...


class RunTraceReader(ABC):
    @abstractmethod
    def read_run(self, run_id: str, tenant_id: str) -> PersistedRun:
        """Implementations MUST enforce filtering by both run_id and tenant_id."""

    def list_runs(self, tenant_id: str, *, limit: int = 50) -> list[RunSummary]:
        raise NotImplementedError(f"{type(self).__name__} does not support list_runs")


class RunTraceStore(RunTraceWriter, RunTraceReader):
    """Canonical persisted trace store with read and write capabilities."""


__all__ = [
    "RunTraceReader",
    "RunTraceStore",
    "RunTraceWriter",
]
