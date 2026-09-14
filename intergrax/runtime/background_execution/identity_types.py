# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared background execution identity durable types."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)


@dataclass(frozen=True, slots=True)
class PersistedBackgroundExecutionIdentity:
    """Durable canonical TaskId/RunId/AttemptId/ExecutionId for one transport execution."""

    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
