# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator-facing diagnostic read DTOs (DIAG-6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessment
from intergrax.runtime.diagnostics.deterministic_problem_reconciliation import (
    DeterministicProblemReconciliationKey,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicProblemSignature,
    ProblemGroupingMethod,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyVersion,
    ProblemGroupingSubjectRef,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    ProblemId,
    ProblemLifecycleProvenance,
    ProblemOccurrenceAggregateHealth,
    ProblemReconciliationKeyKind,
    ProblemStatus,
)


class DiagnosticReadIntegrityError(Exception):
    """Raised when persisted Problem data or reconstruction scope is structurally inconsistent."""


class DiagnosticReadUnavailableReason(StrEnum):
    """Expected absence of canonical execution evidence — not structural corruption."""

    EXECUTION_EVIDENCE_UNAVAILABLE = "execution_evidence_unavailable"
    NON_EXECUTION_SUBJECT = "non_execution_subject"


class DiagnosticOccurrenceReadStatus(StrEnum):
    """Whether diagnostic assessment was reconstructed for one occurrence."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class DiagnosticGroupingProvenance:
    """Typed grouping provenance for operator audit — no internal index tokens."""

    strategy_id: ProblemGroupingStrategyId
    strategy_version: ProblemGroupingStrategyVersion
    method: ProblemGroupingMethod
    reconciliation_key_kind: ProblemReconciliationKeyKind
    deterministic_signature: DeterministicProblemSignature | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticProblemSummary:
    """Cheap list item — no occurrence history or reconstructed diagnostics."""

    problem_id: ProblemId
    tenant_id: str
    status: ProblemStatus
    first_seen_at: datetime
    last_seen_at: datetime
    occurrence_count: int
    grouping_provenance: DiagnosticGroupingProvenance
    occurrence_aggregate_health: ProblemOccurrenceAggregateHealth


@dataclass(frozen=True, slots=True)
class DiagnosticProblemListResult:
    problems: tuple[DiagnosticProblemSummary, ...]
    total_count: int | None
    returned_count: int
    is_truncated: bool
    has_more: bool
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class DiagnosticProblemOccurrenceView:
    subject_ref: ProblemGroupingSubjectRef
    observed_at: datetime
    strategy_id: ProblemGroupingStrategyId
    strategy_version: ProblemGroupingStrategyVersion
    method: ProblemGroupingMethod
    read_status: DiagnosticOccurrenceReadStatus
    assessment: DiagnosticAssessment | None
    unavailable_reason: DiagnosticReadUnavailableReason | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticProblemDetail:
    problem_id: ProblemId
    tenant_id: str
    status: ProblemStatus
    first_seen_at: datetime
    last_seen_at: datetime
    occurrence_count: int
    record_version: int
    grouping_provenance: DiagnosticGroupingProvenance
    occurrence_aggregate_health: ProblemOccurrenceAggregateHealth
    occurrences: tuple[DiagnosticProblemOccurrenceView, ...]
    returned_occurrence_count: int
    total_occurrence_count: int
    is_occurrences_truncated: bool


def grouping_provenance_from_problem_provenance(
    provenance: ProblemLifecycleProvenance,
) -> DiagnosticGroupingProvenance:
    """Map persisted lifecycle provenance to operator-safe grouping provenance."""
    if type(provenance) is not ProblemLifecycleProvenance:
        raise TypeError("provenance must be ProblemLifecycleProvenance")

    reconciliation_key = provenance.reconciliation_key
    deterministic_signature: DeterministicProblemSignature | None = None
    if type(reconciliation_key) is DeterministicProblemReconciliationKey:
        deterministic_signature = reconciliation_key.signature

    return DiagnosticGroupingProvenance(
        strategy_id=provenance.strategy_id,
        strategy_version=provenance.strategy_version,
        method=provenance.method,
        reconciliation_key_kind=reconciliation_key.kind,
        deterministic_signature=deterministic_signature,
    )
