# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Contracts for multi-agent execution failure localization (DIAG R3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_identity import EventId, ExecutionId


class FailureBoundaryPrecision(StrEnum):
    """Localization granularity for a failure boundary (aligned with diagnostic precision)."""

    RUN_LEVEL = "run_level"
    ATTEMPT_LEVEL = "attempt_level"
    EXECUTION_LEVEL = "execution_level"
    EXTERNAL_BOUNDARY = "external_boundary"


class FailureBoundaryCertainty(StrEnum):
    """Certainty for a localized execution failure boundary (R3)."""

    PROVEN = "proven"
    SUPPORTED = "supported"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


class DiagnosticFailureTopologyCompleteness(StrEnum):
    """Whether impact / healthy scope could be fully derived from lineage."""

    COMPLETE = "complete"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class DiagnosticFailureBoundary:
    """Proven or bounded failure localization at execution granularity."""

    execution_id: ExecutionId
    precision: FailureBoundaryPrecision
    certainty: FailureBoundaryCertainty
    evidence_refs: tuple[EventId, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticFailureTopology:
    """Impact grouping for one attempt scope — lineage impact, not causality."""

    failed_boundaries: tuple[DiagnosticFailureBoundary, ...]
    affected_executions: tuple[ExecutionId, ...]
    healthy_executions: tuple[ExecutionId, ...]
    unknown_scope: tuple[ExecutionId, ...]
    completeness: DiagnosticFailureTopologyCompleteness


@dataclass(frozen=True, slots=True)
class FailureBoundaryAnalysis:
    """
    Derived failure localization for an execution reconstruction scope.

    Does not attribute root cause; parent lineage edges are impact only.
    """

    boundaries: tuple[DiagnosticFailureBoundary, ...]
    topology: DiagnosticFailureTopology

    @staticmethod
    def unavailable() -> FailureBoundaryAnalysis:
        topology = DiagnosticFailureTopology(
            failed_boundaries=(),
            affected_executions=(),
            healthy_executions=(),
            unknown_scope=(),
            completeness=DiagnosticFailureTopologyCompleteness.UNAVAILABLE,
        )
        return FailureBoundaryAnalysis(boundaries=(), topology=topology)


__all__ = [
    "DiagnosticFailureBoundary",
    "DiagnosticFailureTopology",
    "DiagnosticFailureTopologyCompleteness",
    "FailureBoundaryAnalysis",
    "FailureBoundaryCertainty",
    "FailureBoundaryPrecision",
]
