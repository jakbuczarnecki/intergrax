# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)


def runtime_event_test_identity(
    *,
    task_id: TaskId | str | None = None,
    run_id: RunId | str | None = None,
    attempt_id: AttemptId | str | None = None,
    execution_id: ExecutionId | str | None = None,
) -> dict[str, TaskId | RunId | AttemptId | ExecutionId]:
    return {
        "task_id": TaskId(task_id) if task_id is not None else mint_task_id(),
        "run_id": RunId(run_id) if run_id is not None else mint_run_id(),
        "attempt_id": AttemptId(attempt_id) if attempt_id is not None else mint_attempt_id(),
        "execution_id": ExecutionId(execution_id)
        if execution_id is not None
        else mint_execution_id(),
    }


def emit_context_test_identity(
    *,
    task_id: TaskId | str | None = None,
    run_id: RunId | str | None = None,
    attempt_id: AttemptId | str | None = None,
    execution_id: ExecutionId | str | None = None,
    tenant_id: str | None = None,
    correlation_id: str = "",
    parent_event_id: EventId | None = None,
    traceparent: str | None = None,
    tracestate: str | None = None,
    bus: object | None = None,
    production_mode: bool = False,
) -> "EmitContext":
    from intergrax.runtime.events.emit_context import EmitContext
    from intergrax.runtime.events.event_bus import RuntimeEventBus

    identity = runtime_event_test_identity(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    return EmitContext(
        task_id=identity["task_id"],
        run_id=identity["run_id"],
        attempt_id=identity["attempt_id"],
        execution_id=identity["execution_id"],
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        parent_event_id=parent_event_id,
        traceparent=traceparent,
        tracestate=tracestate,
        bus=bus if isinstance(bus, RuntimeEventBus) else None,
        production_mode=production_mode,
    )
