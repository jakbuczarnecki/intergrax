# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical active execution → TaskId scope authority port."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)


class ActiveExecutionTaskScopeUnavailable(RuntimeError):
    """No canonical TaskId correlates with the active execution tree."""


class ActiveExecutionTaskScopePort(Protocol):
    """Resolve owning TaskId for the currently active execution context."""

    def resolve_current_task_scope(
        self,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> TaskId: ...


__all__ = [
    "ActiveExecutionTaskScopePort",
    "ActiveExecutionTaskScopeUnavailable",
]
