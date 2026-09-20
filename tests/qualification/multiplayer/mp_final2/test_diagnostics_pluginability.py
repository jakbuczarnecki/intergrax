# © Artur Czarnecki. All rights reserved.

"""MP-FINAL-2 — pluginability of Evidence / Diagnostics seams without Multiplayer source changes."""

from __future__ import annotations

import pytest

from intergrax.contracts.collaborative_decision_binding import CollaborativeDecisionBinding
from intergrax.contracts.collaborative_functional_evidence_projection import (
    CollaborativeDecisionBindingCreateOutcomeProjection,
    CollaborativeFunctionalEvidenceProjectionStrategy,
)
from intergrax.contracts.functional_evidence.correlation import (
    FunctionalEvidenceExecutionCorrelation,
)
from intergrax.contracts.functional_evidence.models import (
    PipelineEvidenceKind,
    PipelineOperationStatus,
    PlatformFunctionalEvidence,
)
from intergrax.contracts.functional_evidence.persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidenceQueryPage,
    FunctionalEvidenceQueryRequest,
)
from intergrax.runtime.diagnostics.functional_operator_projection import (
    FunctionalOperatorOutcomeStatus,
)
from tests.qualification.multiplayer.mp_final2.host_operability import (
    FailingCreateBindingRepository,
    binding_create_request,
    build_operability_host,
    execution_correlation_for,
    interpret_binding_create_operability,
)

pytestmark = [pytest.mark.unit, pytest.mark.qualification]


class _RecordingEvidencePersistence(FunctionalEvidencePersistence):
    """Custom conforming FunctionalEvidencePersistence (public contract)."""

    def __init__(self) -> None:
        self.appended: list[PlatformFunctionalEvidence] = []
        self._items: dict[str, PlatformFunctionalEvidence] = {}

    def append(self, evidence: PlatformFunctionalEvidence) -> PlatformFunctionalEvidence:
        key = str(evidence.evidence_id)
        existing = self._items.get(key)
        if existing is not None:
            return existing
        self._items[key] = evidence
        self.appended.append(evidence)
        return evidence

    def query_evidence(self, request: FunctionalEvidenceQueryRequest) -> FunctionalEvidenceQueryPage:
        items = tuple(
            item
            for item in self.appended
            if item.scope.tenant_id == request.tenant_id
            and item.scope.task_id == request.task_id
            and item.scope.run_id == request.run_id
            and (request.attempt_id is None or item.scope.attempt_id == request.attempt_id)
            and (request.kind is None or item.kind is request.kind)
        )
        return FunctionalEvidenceQueryPage(
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            run_id=request.run_id,
            items=items,
            next_cursor=None,
        )


class _TaggedProjectionStrategy(CollaborativeFunctionalEvidenceProjectionStrategy):
    """Replaceable projection strategy — translation only."""

    def __init__(self, delegate: CollaborativeFunctionalEvidenceProjectionStrategy) -> None:
        self.delegate = delegate
        self.calls = 0

    def project_decision_binding_create_outcome(
        self,
        projection: CollaborativeDecisionBindingCreateOutcomeProjection,
    ) -> PlatformFunctionalEvidence:
        self.calls += 1
        return self.delegate.project_decision_binding_create_outcome(projection)

    def project_decision_binding_association(
        self,
        binding: CollaborativeDecisionBinding,
        *,
        execution_correlation: FunctionalEvidenceExecutionCorrelation | None,
    ) -> tuple[PlatformFunctionalEvidence, ...]:
        return self.delegate.project_decision_binding_association(
            binding,
            execution_correlation=execution_correlation,
        )


def test_custom_evidence_persistence_is_injectable_without_source_changes() -> None:
    from intergrax.collaborative_work.functional_evidence_projection import (
        DefaultCollaborativeFunctionalEvidenceProjection,
    )

    persistence = _RecordingEvidencePersistence()
    strategy = _TaggedProjectionStrategy(DefaultCollaborativeFunctionalEvidenceProjection())
    host = build_operability_host(
        binding_repository=FailingCreateBindingRepository(),
        evidence_persistence=persistence,
        projection_strategy=strategy,
    )

    with pytest.raises(RuntimeError):
        host.application.create_binding(
            binding_create_request(host),
            execution_correlation=execution_correlation_for(host),
        )

    assert strategy.calls == 1
    assert len(persistence.appended) == 1
    assert persistence.appended[0].operation_outcome is not None
    assert persistence.appended[0].operation_outcome.status is PipelineOperationStatus.FAILED
    assert persistence.appended[0].kind is PipelineEvidenceKind.OPERATION_OUTCOME

    projection = interpret_binding_create_operability(host)
    assert projection.outcome_status is FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE


def test_pluginability_matrix_contracts_are_protocols_or_abc() -> None:
    """Documented replaceable seams remain contract-backed."""
    assert issubclass(FunctionalEvidencePersistence, object)
    assert hasattr(FunctionalEvidencePersistence, "append")
    assert hasattr(FunctionalEvidencePersistence, "query_evidence")
    assert issubclass(CollaborativeFunctionalEvidenceProjectionStrategy, object)
    assert hasattr(CollaborativeFunctionalEvidenceProjectionStrategy, "project_decision_binding_create_outcome")
