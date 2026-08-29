# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.execution_identity import (
    AttemptId,
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
