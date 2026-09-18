# © Artur Czarnecki. All rights reserved.

"""Test-only helpers for authoritative RuntimeEventMetricScope (EXEC-R4)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event_metric_scope import RuntimeEventMetricScope


def open_runtime_event_metric_scope_for_tests(
    *,
    task_id: TaskId,
    run_id: RunId,
) -> RuntimeEventMetricScope:
    """Open a real per-invocation metric scope; caller must ``close()`` it."""
    return RuntimeEventBus().open_runtime_event_metric_scope(task_id, run_id)
