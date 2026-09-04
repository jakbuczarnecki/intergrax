# © Artur Czarnecki. All rights reserved.

"""Canonical resume admission gate for checkpoint-backed execution (P0C-5)."""

from __future__ import annotations

from intergrax.contracts.execution_terminal import (
    ExecutionTerminalError,
    ExecutionTerminalOutcome,
)
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task_state import TaskState

_RESUMABLE_CHECKPOINT_STATES = frozenset(
    {
        TaskState.WAITING_FOR_HUMAN,
        TaskState.WAITING_FOR_RESOURCES,
        TaskState.NEEDS_MORE_INFORMATION,
    }
)

TERMINALLY_CANCELLED_RESUME_MSG = (
    "task execution is terminally cancelled and cannot be resumed"
)


class CheckpointNotResumableError(RuntimeError):
    """Fail-closed resume denial for non-resumable checkpoint-backed work."""


def is_checkpoint_resumable(
    checkpoint: TaskCheckpoint,
    *,
    execution_terminal: ExecutionTerminalService | None = None,
) -> bool:
    try:
        assert_checkpoint_resumable(checkpoint, execution_terminal=execution_terminal)
    except CheckpointNotResumableError:
        return False
    return True


def assert_checkpoint_resumable(
    checkpoint: TaskCheckpoint,
    *,
    execution_terminal: ExecutionTerminalService | None = None,
) -> None:
    if checkpoint.task_state not in _RESUMABLE_CHECKPOINT_STATES:
        raise CheckpointNotResumableError(
            f"checkpoint state {checkpoint.task_state.value!r} is not resumable",
        )
    if execution_terminal is None:
        return
    try:
        record = execution_terminal.get_terminal_record(
            tenant_id=checkpoint.tenant_id,
            task_id=checkpoint.task_id,
        )
    except ExecutionTerminalError as exc:
        raise CheckpointNotResumableError(
            "execution terminal authority is corrupt or unavailable",
        ) from exc
    if record is None:
        return
    if record.outcome is ExecutionTerminalOutcome.CANCELLED:
        raise CheckpointNotResumableError(TERMINALLY_CANCELLED_RESUME_MSG)
    raise CheckpointNotResumableError(
        f"task execution reached terminal outcome {record.outcome.value!r}",
    )
