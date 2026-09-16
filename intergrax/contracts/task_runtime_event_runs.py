# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Task-scoped runtime evidence grouped by run (neutral read DTO)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import RunId
from intergrax.contracts.positioned_runtime_event import PositionedRuntimeEvent


@dataclass(frozen=True, slots=True)
class TaskRuntimeEventRuns:
    """
    Task-scoped events grouped by run.

    Each run group is ordered by ``ExecutionEventPosition`` (run-local only).
    Run groups are ordered by canonical ``run_id`` — not task-global chronology.
    """

    runs: tuple[tuple[RunId, tuple[PositionedRuntimeEvent, ...]], ...]


__all__ = ["TaskRuntimeEventRuns"]
