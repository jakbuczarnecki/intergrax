# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical four-ID execution binding for boundary and continuation enforcement."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)


@dataclass(frozen=True, slots=True)
class ExecutionIdentityBinding:
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    parent_execution_id: ExecutionId | None = None
    task_id: TaskId | None = None


__all__ = ["ExecutionIdentityBinding"]
