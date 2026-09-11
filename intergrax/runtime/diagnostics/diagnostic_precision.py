# © Artur Czarnecki. All rights reserved.

"""Diagnostic certainty, precision, and failure boundary semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_identity import EventId, ExecutionId


class DiagnosticCertainty(StrEnum):
    """Semantic certainty for operator-facing diagnostic claims."""

    PROVEN = "proven"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DiagnosticPrecision(StrEnum):
    RUN_LEVEL = "run_level"
    ATTEMPT_LEVEL = "attempt_level"
    EXECUTION_LEVEL = "execution_level"
    EXTERNAL_BOUNDARY = "external_boundary"


@dataclass(frozen=True, slots=True)
class FailureBoundary:
    execution_id: ExecutionId
    supporting_event_id: EventId
    certainty: DiagnosticCertainty
    precision: DiagnosticPrecision


__all__ = [
    "DiagnosticCertainty",
    "DiagnosticPrecision",
    "FailureBoundary",
]
