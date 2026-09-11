# © Artur Czarnecki. All rights reserved.

"""Canonical completion reconciliation phase diagnostic (DS-E2E-15J-O1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


class ReconciliationPhaseValue(StrEnum):
    """Runtime-proven reconciliation phase transitions."""

    ENTERED = "entered"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ReconciliationPhaseDiagV1(DiagnosticPayload):
    """
    Typed reconciliation boundary evidence.

    ``entered_reconciliation`` is True only after runtime enters reconciliation.
    Pre-reconciliation validation rejection must leave this False.
    """

    run_id: str
    attempt_index: int
    validation_invalid: bool
    entered_reconciliation: bool
    phase: ReconciliationPhaseValue

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.completion.reconciliation_phase.v1"

    @classmethod
    def schema_version(cls) -> int:
        return 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "attempt_index": self.attempt_index,
            "validation_invalid": self.validation_invalid,
            "entered_reconciliation": self.entered_reconciliation,
            "phase": self.phase.value,
        }

    def redact(self) -> ReconciliationPhaseDiagV1:
        return self
