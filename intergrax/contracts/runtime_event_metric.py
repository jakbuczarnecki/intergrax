# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Task/run-scoped RuntimeEventBus acceptance metrics (non-canonical)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.execution_identity import RunId, TaskId


class RuntimeEventMetricScope(Protocol):
    """Opaque per-invocation counter for events accepted on the bus for one task/run."""

    def count(self) -> int: ...

    def close(self) -> None: ...


class RuntimeEventMetricScopeFactory(Protocol):
    """Opens a scoped counter keyed by canonical task/run identity."""

    def open_runtime_event_metric_scope(
        self,
        task_id: TaskId,
        run_id: RunId,
    ) -> RuntimeEventMetricScope: ...


__all__ = [
    "RuntimeEventMetricScope",
    "RuntimeEventMetricScopeFactory",
]
