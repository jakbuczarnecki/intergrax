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
from intergrax.runtime.nexus.tracing.persistence_models import RunMetadata, RunStats, RunTraceWriter
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


class PersistingTaskTraceEmitter(TaskTraceEmitter):
    """TaskTraceEmitter that writes events to a RunTraceWriter (Phase A.2)."""

    def __init__(
        self,
        *,
        run_id: str,
        trace_store: RunTraceWriter,
        tenant_id: str,
        user_id: str,
        session_id: str = "",
    ) -> None:
        super().__init__(run_id=run_id)
        self._trace_store = trace_store
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._session_id = session_id
        self._started_at_utc: Optional[str] = None

    def emit(self, task: Task, *, message: str) -> TraceEvent:
        evt = super().emit(task, message=message)
        self._trace_store.append_event(evt)
        if self._started_at_utc is None:
            self._started_at_utc = evt.ts_utc
        return evt

    def finalize(self, *, duration_ms: int = 0) -> None:
        if self._started_at_utc is None:
            return
        self._trace_store.finalize_run(
            self._run_id,
            RunMetadata(
                run_id=self._run_id,
                session_id=self._session_id,
                user_id=self._user_id,
                tenant_id=self._tenant_id,
                started_at_utc=self._started_at_utc,
                stats=RunStats(duration_ms=duration_ms, llm_usage={}),
            ),
        )


def lifecycle_with_persisting_trace(
    *,
    run_id: str,
    trace_store: RunTraceWriter,
    tenant_id: str,
    user_id: str,
    session_id: str = "",
    on_transition: Optional[Callable[[Task], None]] = None,
) -> tuple[TaskLifecycle, PersistingTaskTraceEmitter]:
    emitter = PersistingTaskTraceEmitter(
        run_id=run_id,
        trace_store=trace_store,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
    )

    def _combined(task: Task) -> None:
        emitter.as_transition_handler()(task)
        if on_transition is not None:
            on_transition(task)

    return TaskLifecycle(on_transition=_combined), emitter


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
