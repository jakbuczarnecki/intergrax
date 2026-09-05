# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime port for effective profile revision admission (P1.2A)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task


class EffectiveProfileRevisionAdmissionPort(Protocol):
    """Fail-closed admission gate before meaningful host task execution."""

    def admit_root_execution(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        task: Task,
        resume_checkpoint: TaskCheckpoint | None = None,
        restore_existing_execution: bool = False,
    ) -> Task:
        """Pin or verify revision binding; return task with checkpoint evidence."""
