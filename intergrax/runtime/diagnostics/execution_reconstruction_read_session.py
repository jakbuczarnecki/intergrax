# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Request-scoped execution reconstruction reuse for diagnostic reads (DIAG-READ-SCALE)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_event_position import AsOfBoundary
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.contracts.execution_reconstruction import (
    ExecutionReconstruction,
    ExecutionReconstructionReader,
)


@dataclass(frozen=True, slots=True)
class ExecutionReconstructionScopeKey:
    """Identity for one factual reconstruction read (tenant + run + optional as-of)."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    execution_as_of: AsOfBoundary | None


class ExecutionReconstructionReadSession:
    """
    Bounded request-local reuse over ``ExecutionReconstructionReader``.

    Memoizes successful ``ExecutionReconstruction`` results only for the lifetime of
    this session. Not a source of truth and not shared across operator read requests.
    """

    def __init__(self, reader: ExecutionReconstructionReader) -> None:
        self._reader = reader
        self._memo: dict[ExecutionReconstructionScopeKey, ExecutionReconstruction] = {}

    def reconstruct_execution(
        self,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        *,
        execution_as_of: AsOfBoundary | None = None,
    ) -> ExecutionReconstruction:
        key = ExecutionReconstructionScopeKey(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            execution_as_of=execution_as_of,
        )
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        reconstruction = self._reader.reconstruct_execution(
            tenant_id,
            task_id,
            run_id,
            execution_as_of=execution_as_of,
        )
        self._memo[key] = reconstruction
        return reconstruction

    def memo_entry_count(self) -> int:
        """Number of unique scopes memoized in this session (structural bound for tests)."""
        return len(self._memo)
