# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

from intergrax.contracts.execution_identity import AttemptId, RunId

if TYPE_CHECKING:
    from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.tracing.trace_models import (
    DiagnosticPayload,
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

    When ``event_bus`` is provided, also records canonical ``RuntimeEvent`` entries
    via ``trace_bridge`` (§42.1, P4.1) — without replacing trace persistence.
    """

    def __init__(
        self,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        event_bus: Optional[RuntimeEventBus] = None,
    ) -> None:
        self._run_id = run_id
        self._attempt_id = attempt_id
        self._seq = 0
        self.events: List[TraceEvent] = []
        self._event_bus = event_bus

    @property
    def event_bus(self) -> Optional[RuntimeEventBus]:
        return self._event_bus

    def emit_trace_step(
        self,
        task: Task,
        *,
        component: TraceComponent,
        step: str,
        message: str,
        level: TraceLevel = TraceLevel.INFO,
        payload: DiagnosticPayload | None = None,
        extra_tags: dict[str, object] | None = None,
    ) -> TraceEvent:
        if payload is not None:
            payload = payload.redact()
        self._seq += 1
        tags: dict[str, object] = {
            "task_id": task.task_id,
            "task_state": task.state.value,
            "agent_id": task.agent_id,
            "capability": task.context.capability,
            "tenant_id": task.tenant_id,
        }
        if extra_tags:
            tags.update(extra_tags)
        evt = TraceEvent(
            event_id=TraceEvent.new_id(),
            run_id=self._run_id,
            seq=self._seq,
            ts_utc=utc_now_iso(),
            level=level,
            component=component,
            step=step,
            message=message,
            payload=payload,
            tags=tags,
        )
        self.events.append(evt)
        if self._event_bus is not None:
            from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event

            self._event_bus.record(
                trace_event_to_runtime_event(
                    evt,
                    task,
                    run_id=self._run_id,
                    attempt_id=self._attempt_id,
                )
            )
        return evt

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
        if self._event_bus is not None:
            from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event

            self._event_bus.record(
                trace_event_to_runtime_event(
                    evt,
                    task,
                    run_id=self._run_id,
                    attempt_id=self._attempt_id,
                )
            )
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
        run_id: RunId,
        attempt_id: AttemptId,
        trace_store: RunTraceWriter,
        tenant_id: str,
        user_id: str,
        session_id: str = "",
        event_bus: Optional[RuntimeEventBus] = None,
    ) -> None:
        super().__init__(run_id=run_id, attempt_id=attempt_id, event_bus=event_bus)
        self._trace_store = trace_store
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._session_id = session_id
        self._started_at_utc: Optional[str] = None

    @property
    def trace_store(self) -> RunTraceWriter:
        return self._trace_store

    def emit(self, task: Task, *, message: str) -> TraceEvent:
        evt = super().emit(task, message=message)
        self._trace_store.append_event(evt)
        if self._started_at_utc is None:
            self._started_at_utc = evt.ts_utc
        return evt

    def emit_trace_step(
        self,
        task: Task,
        *,
        component: TraceComponent,
        step: str,
        message: str,
        level: TraceLevel = TraceLevel.INFO,
        payload: DiagnosticPayload | None = None,
        extra_tags: dict[str, object] | None = None,
    ) -> TraceEvent:
        evt = super().emit_trace_step(
            task,
            component=component,
            step=step,
            message=message,
            level=level,
            payload=payload,
            extra_tags=extra_tags,
        )
        self._trace_store.append_event(evt)
        if self._started_at_utc is None:
            self._started_at_utc = evt.ts_utc
        return evt

    def finalize(
        self,
        *,
        duration_ms: int = 0,
        llm_usage: dict[str, object] | None = None,
    ) -> None:
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
                stats=RunStats(
                    duration_ms=duration_ms,
                    llm_usage=dict(llm_usage or {}),
                ),
            ),
        )


def lifecycle_with_persisting_trace(
    *,
    run_id: RunId,
    attempt_id: AttemptId,
    trace_store: RunTraceWriter,
    tenant_id: str,
    user_id: str,
    session_id: str = "",
    event_bus: Optional[RuntimeEventBus] = None,
    on_transition: Optional[Callable[[Task], None]] = None,
) -> tuple[TaskLifecycle, PersistingTaskTraceEmitter]:
    emitter = PersistingTaskTraceEmitter(
        run_id=run_id,
        attempt_id=attempt_id,
        trace_store=trace_store,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        event_bus=event_bus,
    )

    def _combined(task: Task) -> None:
        emitter.as_transition_handler()(task)
        if on_transition is not None:
            on_transition(task)

    return TaskLifecycle(on_transition=_combined), emitter


def lifecycle_with_trace(
    run_id: RunId,
    attempt_id: AttemptId,
    *,
    event_bus: Optional[RuntimeEventBus] = None,
    on_transition: Optional[Callable[[Task], None]] = None,
) -> tuple[TaskLifecycle, TaskTraceEmitter]:
    """Build TaskLifecycle wired to a TaskTraceEmitter."""
    emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_id, event_bus=event_bus)

    def _combined(task: Task) -> None:
        emitter.as_transition_handler()(task)
        if on_transition is not None:
            on_transition(task)

    return TaskLifecycle(on_transition=_combined), emitter
