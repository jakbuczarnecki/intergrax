# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only factual execution reconstruction port (OBS-DIAG-CONFORMANCE-R1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_event_position import AsOfBoundary
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.contracts.execution_reconstruction_models import ExecutionReconstruction

__all__ = [
    "ExecutionAttemptDiscoveryCompleteness",
    "ExecutionAttemptDiscoveryReadStatus",
    "ExecutionReconstruction",
    "ExecutionReconstructionIntegrityError",
    "ExecutionReconstructionReader",
    "ReconstructedAttempt",
    "RuntimeHistoryCompleteness",
]

from intergrax.contracts.execution_reconstruction_models import (  # noqa: E402
    ExecutionAttemptDiscoveryCompleteness,
    ExecutionAttemptDiscoveryReadStatus,
    ExecutionReconstructionIntegrityError,
    ReconstructedAttempt,
    RuntimeHistoryCompleteness,
)


@runtime_checkable
class ExecutionReconstructionReader(Protocol):
    """
    Read-only provider-neutral boundary for canonical factual execution reconstruction.

    Implementations MUST preserve platform reconstruction semantics and identity integrity.
    """

    def reconstruct_execution(
        self,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        *,
        execution_as_of: AsOfBoundary | None = None,
    ) -> ExecutionReconstruction:
        """Rebuild derived execution facts for one tenant-scoped run."""
