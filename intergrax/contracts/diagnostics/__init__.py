# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Diagnostic domain contracts (persistence ports, read models, operator projection)."""

from __future__ import annotations

from intergrax.contracts.diagnostics.diagnostic_read_model import (
    PersistedProblem,
    ProblemListPage,
)
from intergrax.contracts.diagnostics.diagnostic_repository import (
    DiagnosticProblemRepository,
)
from intergrax.contracts.diagnostics.functional_diagnostic_check_status import (
    FunctionalDiagnosticCheckStatus,
)
from intergrax.contracts.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
    validate_functional_diagnostic_check_id,
    validate_functional_diagnostic_specification_id,
    validate_functional_diagnostic_specification_version,
)
from intergrax.contracts.diagnostics.functional_operator_projection import (
    FunctionalCheckPassResult,
    FunctionalDiagnosticOperatorFinding,
    FunctionalDiagnosticOperatorLimitation,
    FunctionalDiagnosticOperatorProjection,
    FunctionalDiagnosticSummary,
    FunctionalOperatorOutcomeStatus,
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
from intergrax.contracts.diagnostics.terminal_execution_diagnostic_port import (
    TerminalDiagnosticDispatchResult,
    TerminalDiagnosticDispatchStatus,
    TerminalExecutionDiagnosticPort,
    TerminalExecutionDiagnosticRequest,
)

__all__ = [
    "DiagnosticProblemRepository",
    "FunctionalCheckPassResult",
    "FunctionalDiagnosticCheckId",
    "FunctionalDiagnosticCheckStatus",
    "FunctionalDiagnosticOperatorFinding",
    "FunctionalDiagnosticOperatorLimitation",
    "FunctionalDiagnosticOperatorProjection",
    "FunctionalDiagnosticSpecificationId",
    "FunctionalDiagnosticSummary",
    "FunctionalOperatorOutcomeStatus",
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
    "TerminalDiagnosticDispatchResult",
    "TerminalDiagnosticDispatchStatus",
    "TerminalExecutionDiagnosticPort",
    "TerminalExecutionDiagnosticRequest",
    "validate_functional_diagnostic_check_id",
    "validate_functional_diagnostic_specification_id",
    "validate_functional_diagnostic_specification_version",
]
