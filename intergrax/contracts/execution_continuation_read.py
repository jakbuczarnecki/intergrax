# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only execution continuation snapshot access (SESSION-01 read seam)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationIdentity,
    PendingExecutionContinuation,
)


@runtime_checkable
class ExecutionContinuationSnapshotReadPort(Protocol):
    """Reads canonical current continuation episode — no pause/resume commands."""

    @property
    def source_id(self) -> str: ...

    @property
    def is_durable(self) -> bool: ...

    def read_current_episode(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None: ...


__all__ = ["ExecutionContinuationSnapshotReadPort"]
