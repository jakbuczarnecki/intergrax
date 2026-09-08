# © Artur Czarnecki. All rights reserved.

"""Neutral reusable execution provenance reference (Unified Execution linkage)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)


@dataclass(frozen=True, slots=True)
class ExecutionProvenanceRef:
    """Four-part canonical execution provenance — all fields required."""

    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId

    def __post_init__(self) -> None:
        validate_task_id(self.task_id)
        validate_run_id(self.run_id)
        validate_attempt_id(self.attempt_id)
        validate_execution_id(self.execution_id)
