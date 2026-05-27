# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Callable, List, Optional

from intergrax.runtime.nexus.tracing.trace_models import (
    TraceComponent,
    TraceEvent,
    TraceLevel,
    utc_now_iso,
)
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task import Task


class TaskTraceEmitter:
    """
    Emits TraceEvent records for TaskLifecycle transitions (§23).
    """

    def __init__(self, *, run_id: str) -> None:
        self._run_id = run_id
        self._seq = 0
        self.events: List[TraceEvent] = []

    def emit(self, task: Task, *, message: str) -> TraceEvent:
        self._seq += 1
        evt = TraceEvent(
            event_id=TraceEvent.new_id(),
            run_id=self._run_id,
            seq=self._seq,
            ts_utc=utc_now_iso(),
            level=TraceLevel.INFO,
            component=TraceComponent.PLANNER,
            step="task_lifecycle",
            message=message,
            tags={
                "task_id": task.task_id,
                "task_state": task.state.value,
                "agent_id": task.agent_id,
                "capability": task.context.capability,
            },
        )
        self.events.append(evt)
        return evt

    def as_transition_handler(self) -> Callable[[Task], None]:
        def _handler(task: Task) -> None:
            self.emit(task, message=f"task state -> {task.state.value}")

        return _handler


def lifecycle_with_trace(
    run_id: str,
    *,
    on_transition: Optional[Callable[[Task], None]] = None,
) -> tuple[TaskLifecycle, TaskTraceEmitter]:
    """Build TaskLifecycle wired to a TaskTraceEmitter."""
    from intergrax.runtime.task.task_lifecycle import TaskLifecycle

    emitter = TaskTraceEmitter(run_id=run_id)

    def _combined(task: Task) -> None:
        emitter.as_transition_handler()(task)
        if on_transition is not None:
            on_transition(task)

    return TaskLifecycle(on_transition=_combined), emitter
