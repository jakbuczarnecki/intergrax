# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Diagnostic domain contracts (persistence ports and read models)."""

from __future__ import annotations

from intergrax.contracts.diagnostics.diagnostic_read_model import (
    PersistedProblem,
    ProblemListPage,
)
from intergrax.contracts.diagnostics.diagnostic_repository import (
    DiagnosticProblemRepository,
)
from intergrax.contracts.diagnostics.problem_identity import (
    ProblemId,
    ProblemOccurrenceAggregateHealth,
    ProblemStatus,
)
from intergrax.contracts.diagnostics.problem_persistence import (
    ProblemPersistence,
    ProblemPersistenceConflictError,
    ProblemPersistenceIntegrityError,
    ProblemPersistenceIntegrityReason,
)
from intergrax.contracts.diagnostics.reconciliation_key import ProblemReconciliationKey
from intergrax.contracts.diagnostics.subject_ref import ProblemGroupingSubjectRef

__all__ = [
    "DiagnosticProblemRepository",
    "PersistedProblem",
    "ProblemGroupingSubjectRef",
    "ProblemId",
    "ProblemListPage",
    "ProblemOccurrenceAggregateHealth",
    "ProblemPersistence",
    "ProblemPersistenceConflictError",
    "ProblemPersistenceIntegrityError",
    "ProblemPersistenceIntegrityReason",
    "ProblemReconciliationKey",
    "ProblemStatus",
]
