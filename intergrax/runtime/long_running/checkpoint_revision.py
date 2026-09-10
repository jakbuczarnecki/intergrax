# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Logical checkpoint revision CAS types (NPSC-5E/R2-H2)."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

CheckpointRevision = Annotated[int, Field(ge=1)]
"""Monotonic logical revision within one tenant/task checkpoint stream."""


class StaleCheckpointWriteError(RuntimeError):
    """Raised when a checkpoint writer's expected revision does not match canonical state."""

    def __init__(
        self,
        *,
        task_id: str,
        tenant_id: str,
        expected_revision: int | None,
        actual_revision: int | None,
    ) -> None:
        self.task_id = task_id
        self.tenant_id = tenant_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            "stale checkpoint write rejected for "
            f"task_id={task_id!r} tenant_id={tenant_id!r}: "
            f"expected_revision={expected_revision!r}, actual_revision={actual_revision!r}",
        )


class CheckpointRevisionRequiredError(RuntimeError):
    """Raised when an update is attempted without expected_revision."""

    def __init__(self, *, task_id: str, tenant_id: str, actual_revision: int) -> None:
        self.task_id = task_id
        self.tenant_id = tenant_id
        self.actual_revision = actual_revision
        super().__init__(
            "expected_revision is required for checkpoint update on "
            f"task_id={task_id!r} tenant_id={tenant_id!r}; current revision={actual_revision}",
        )


class CheckpointIdConflictError(RuntimeError):
    """Raised when the same checkpoint_id is reused with different payload."""

    def __init__(self, *, checkpoint_id: str, task_id: str, tenant_id: str) -> None:
        self.checkpoint_id = checkpoint_id
        self.task_id = task_id
        self.tenant_id = tenant_id
        super().__init__(
            "checkpoint_id conflict for "
            f"checkpoint_id={checkpoint_id!r} task_id={task_id!r} tenant_id={tenant_id!r}",
        )


__all__ = [
    "CheckpointIdConflictError",
    "CheckpointRevision",
    "CheckpointRevisionRequiredError",
    "StaleCheckpointWriteError",
]
