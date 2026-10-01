# © Artur Czarnecki. All rights reserved.

"""Neutral deferred persisted trace finalize capability (scenario completion alignment)."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import RunId


class PersistedTraceReconciliationPhase(StrEnum):
    """Completion reconciliation phase transitions for deferred trace emission."""

    ENTERED = "entered"
    COMPLETED = "completed"
    FAILED = "failed"


@runtime_checkable
class PersistedTraceCompletionAlignmentPayload(Protocol):
    """Structural completion-alignment diagnostic payload for deferred finalize."""

    @property
    def run_id(self) -> RunId: ...


@runtime_checkable
class DeferredPersistedTraceFinalizePort(Protocol):
    """Deferred persisted trace lifecycle invoked after task completion."""

    def emit_reconciliation_phase_under_identity(
        self,
        *,
        validation_invalid: bool,
        entered_reconciliation: bool,
        phase: PersistedTraceReconciliationPhase,
    ) -> None: ...

    def emit_completion_alignment_under_identity(
        self,
        *,
        payload: PersistedTraceCompletionAlignmentPayload,
    ) -> None: ...


__all__ = [
    "DeferredPersistedTraceFinalizePort",
    "PersistedTraceCompletionAlignmentPayload",
    "PersistedTraceReconciliationPhase",
]
