# © Artur Czarnecki. All rights reserved.

"""Harness for DIAG R4 Decision↔Execution lineage qualification (R4-A1–A7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from intergrax.contracts.decision_execution_correlation import (
    DecisionExecutionCorrelationKind,
    DecisionExecutionCorrelationRecord,
    correlation_record_from_decision_identity,
)
from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.runtime.diagnostics.decision_context_provider import (
    CorrelationBackedDecisionContextProvider,
)
from intergrax.runtime.diagnostics.decision_context_read_models import DecisionContextFact
from intergrax.runtime.diagnostics.decision_execution_correlation_persistence import (
    InMemoryDecisionExecutionCorrelationPersistence,
)
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from testing_support.runtime.execution_failure_evidence_r2_closure_harness import (
    ExecutionFailureEvidenceClosureHarness,
    build_execution_failure_evidence_r2_closure_harness,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    read_service_for_tests,
)

_OBSERVED_AT = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


@dataclass(slots=True)
class DecisionExecutionLineageR4Harness:
    execution: ExecutionFailureEvidenceClosureHarness
    correlation_persistence: InMemoryDecisionExecutionCorrelationPersistence
    decision_context_provider: CorrelationBackedDecisionContextProvider
    contextual_facts_by_decision_id: dict[str, tuple[DecisionContextFact, ...]] = field(
        default_factory=dict,
    )

    @property
    def tenant_id(self) -> str:
        return self.execution.tenant_id

    @property
    def read_service(self) -> DiagnosticReadService:
        return self.execution.read_service

    def append_correlation(
        self,
        identity: DecisionIdentity,
        *,
        kind: DecisionExecutionCorrelationKind = (
            DecisionExecutionCorrelationKind.DECISION_BOUND_EXECUTION
        ),
        created_at: datetime | None = None,
    ) -> DecisionExecutionCorrelationRecord:
        record = correlation_record_from_decision_identity(
            identity,
            correlation_kind=kind,
            created_at=created_at or _OBSERVED_AT,
        )
        self.correlation_persistence.append(record)
        return record

    def bind_contextual_facts(
        self,
        decision_id: str,
        facts: tuple[DecisionContextFact, ...],
    ) -> None:
        self.contextual_facts_by_decision_id[decision_id] = facts
        self.decision_context_provider.contextual_facts_by_decision_id = (
            self.contextual_facts_by_decision_id
        )


def build_decision_execution_lineage_r4_harness(
    *,
    tenant_id: str | None = None,
) -> DecisionExecutionLineageR4Harness:
    execution = build_execution_failure_evidence_r2_closure_harness(tenant_id=tenant_id)
    correlation = InMemoryDecisionExecutionCorrelationPersistence()
    provider = CorrelationBackedDecisionContextProvider(
        correlation_persistence=correlation,
        contextual_facts_by_decision_id={},
    )
    execution.read_service = read_service_for_tests(
        execution.problem_persistence,
        execution.execution_reconstructor,
        occurrence_persistence=execution.occurrence_persistence,
        decision_context_provider=provider,
    )
    return DecisionExecutionLineageR4Harness(
        execution=execution,
        correlation_persistence=correlation,
        decision_context_provider=provider,
    )


__all__ = [
    "DecisionExecutionLineageR4Harness",
    "build_decision_execution_lineage_r4_harness",
]
