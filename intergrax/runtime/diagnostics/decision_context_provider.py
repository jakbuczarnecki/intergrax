# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only Decision context provider for DiagnosticReadService (DIAG R4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.decision_execution_correlation import (
    DecisionExecutionCorrelationRecord,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId
from intergrax.runtime.diagnostics.decision_context_projection import (
    project_decision_context_view,
)
from intergrax.runtime.diagnostics.decision_context_read_models import (
    DecisionContextFact,
    DecisionContextView,
)
from intergrax.runtime.diagnostics.decision_execution_correlation_persistence import (
    DecisionExecutionCorrelationPersistence,
    MAX_CORRELATION_QUERY_RESULTS,
)
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingSubjectRef


class DecisionContextProviderError(Exception):
    """Base failure at decision context read boundary."""


class DecisionContextProviderUnavailableError(DecisionContextProviderError):
    """Decision enrichment backend temporarily unavailable."""


@dataclass(frozen=True, slots=True)
class DecisionContextLookupScope:
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    execution_id: ExecutionId | None


class DecisionContextProvider(Protocol):
    """Optional enrichment — never required for execution diagnostics."""

    def resolve_for_occurrence(
        self,
        *,
        subject_ref: ProblemGroupingSubjectRef,
        contextual_facts_by_decision_id: dict[str, tuple[DecisionContextFact, ...]]
        | None = None,
    ) -> DecisionContextView:
        """Attach related decision context for one occurrence subject."""


@dataclass(slots=True)
class CorrelationBackedDecisionContextProvider:
    """Canonical provider over append-only correlation evidence."""

    correlation_persistence: DecisionExecutionCorrelationPersistence
    unavailable: bool = False
    contextual_facts_by_decision_id: dict[str, tuple[DecisionContextFact, ...]] | None = (
        None
    )

    def resolve_for_occurrence(
        self,
        *,
        subject_ref: ProblemGroupingSubjectRef,
        contextual_facts_by_decision_id: dict[str, tuple[DecisionContextFact, ...]]
        | None = None,
    ) -> DecisionContextView:
        execution = subject_ref.execution()
        if execution is None:
            from intergrax.runtime.diagnostics.decision_context_read_models import (
                DecisionContextUnavailableReason,
            )

            return DecisionContextView.unavailable(
                DecisionContextUnavailableReason.NON_EXECUTION_SUBJECT,
            )
        if self.unavailable:
            raise DecisionContextProviderUnavailableError(
                "decision correlation persistence unavailable",
            )
        records = self.correlation_persistence.query_by_execution_scope(
            tenant_id=subject_ref.tenant_id,
            task_id=execution.task_id,
            run_id=execution.run_id,
            limit=MAX_CORRELATION_QUERY_RESULTS,
        )
        facts = contextual_facts_by_decision_id
        if facts is None:
            facts = self.contextual_facts_by_decision_id
        return project_decision_context_view(
            records,
            contextual_facts_by_decision_id=facts,
        )
