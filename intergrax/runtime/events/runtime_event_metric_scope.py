# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RuntimeEventBus task/run-scoped acceptance counters."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.contracts.runtime_event_metric import RuntimeEventMetricScope


@dataclass(slots=True)
class _ScopeCounter:
    task_id: TaskId
    run_id: RunId
    count: int = 0


@dataclass(slots=True)
class _RuntimeEventMetricScopeHandle(RuntimeEventMetricScope):
    _counter: _ScopeCounter
    _release: Callable[[], None]
    _closed: bool = False

    def count(self) -> int:
        return self._counter.count

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._release()


class _RuntimeEventMetricScopeRegistry:
    """Thread-safe active scopes keyed by canonical task/run identity."""

    __slots__ = ("_lock", "_scopes")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scopes: dict[tuple[TaskId, RunId], _ScopeCounter] = {}

    def open(self, task_id: TaskId, run_id: RunId) -> _RuntimeEventMetricScopeHandle:
        key = (task_id, run_id)
        counter = _ScopeCounter(task_id=task_id, run_id=run_id)
        with self._lock:
            if key in self._scopes:
                raise RuntimeError(
                    "runtime event metric scope already active for task/run",
                )
            self._scopes[key] = counter

        def release() -> None:
            with self._lock:
                self._scopes.pop(key, None)

        return _RuntimeEventMetricScopeHandle(counter, release)

    def record_accepted(self, task_id: TaskId, run_id: RunId) -> None:
        key = (task_id, run_id)
        with self._lock:
            counter = self._scopes.get(key)
            if counter is None:
                return
            counter.count += 1


__all__ = ["_RuntimeEventMetricScopeRegistry"]
