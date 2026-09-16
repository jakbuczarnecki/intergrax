# © Artur Czarnecki. All rights reserved.

"""MP-4R5 — collaborative functional evidence projection behavior."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.collaborative_work.decision_binding_evidence import (
    append_decision_binding_create_outcome_evidence,
)
from intergrax.collaborative_work.functional_evidence_projection import (
    DefaultCollaborativeFunctionalEvidenceProjection,
    execution_correlation_from_provenance,
)
from intergrax.contracts.collaborative_decision_binding import CollaborativeDecisionBinding
from intergrax.contracts.collaborative_functional_evidence_projection import (
    COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID,
    COLLABORATIVE_DECISION_BINDING_EVIDENCE_PRODUCER,
    CollaborativeDecisionBindingAssociationNotRepresentable,
    CollaborativeFunctionalEvidenceNotApplicable,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import DecisionProposalRef, decision_lineage_ref
from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef
from intergrax.contracts.functional_evidence.models import (
    PipelineEvidenceKind,
    PipelineOperationStatus,
    PlatformFunctionalEvidence,
)
from intergrax.contracts.functional_evidence.persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidencePersistenceConflictError,
)
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_SEED = "mp4r5-evidence-projection"
_TASK: TaskId = canonical_task_id_for_tests(_SEED)
_RUN: RunId = canonical_run_id_for_tests(_SEED)
_ATTEMPT: AttemptId = mint_attempt_id()
_EXECUTION: ExecutionId = mint_execution_id()


class _RecordingPersistence(FunctionalEvidencePersistence):
    def __init__(self) -> None:
        self._by_id: dict[str, PlatformFunctionalEvidence] = {}

    def append(self, evidence: PlatformFunctionalEvidence) -> PlatformFunctionalEvidence:
        key = str(evidence.evidence_id)
        existing = self._by_id.get(key)
        if existing is not None:
            if existing.model_dump() != evidence.model_dump():
                raise FunctionalEvidencePersistenceConflictError("conflict")
            return existing
        self._by_id[key] = evidence
        return evidence

    def query_evidence(self, request):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _binding() -> CollaborativeDecisionBinding:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-1"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    proposal = DecisionProposalRef(
        identity=identity,
        lineage_ref=decision_lineage_ref(identity.version),
    )
    return CollaborativeDecisionBinding(
        binding_id="cdb_testbinding000000000000000001",
        tenant_id="tenant-a",
        workspace_id="ws-1",
        work_item_id="wi-1",
        decision_proposal=proposal,
        created_by_principal_id="principal-1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_association_projection_raises_architectural_gap() -> None:
    projector = DefaultCollaborativeFunctionalEvidenceProjection()
    with pytest.raises(CollaborativeDecisionBindingAssociationNotRepresentable):
        projector.project_decision_binding_association(_binding(), execution_correlation=None)


def test_append_without_execution_correlation_is_not_applicable() -> None:
    with pytest.raises(CollaborativeFunctionalEvidenceNotApplicable):
        append_decision_binding_create_outcome_evidence(
            _RecordingPersistence(),
            DefaultCollaborativeFunctionalEvidenceProjection(),
            binding=_binding(),
            operation_status=PipelineOperationStatus.SUCCEEDED,
            execution_correlation=None,
            recorded_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_operation_outcome_uses_exact_execution_correlation() -> None:
    provenance = ExecutionProvenanceRef(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )
    correlation = execution_correlation_from_provenance(tenant_id="tenant-a", execution=provenance)
    evidence_id = mint_event_id()
    recorded_at = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
    persistence = _RecordingPersistence()
    evidence = append_decision_binding_create_outcome_evidence(
        persistence,
        DefaultCollaborativeFunctionalEvidenceProjection(),
        binding=_binding(),
        operation_status=PipelineOperationStatus.SUCCEEDED,
        execution_correlation=correlation,
        recorded_at=recorded_at,
        evidence_id=evidence_id,
    )
    assert evidence.kind is PipelineEvidenceKind.OPERATION_OUTCOME
    assert evidence.scope.task_id == _TASK
    assert evidence.scope.run_id == _RUN
    assert evidence.scope.attempt_id == _ATTEMPT
    assert evidence.scope.execution_id == _EXECUTION
    assert evidence.scope.tenant_id == "tenant-a"
    assert evidence.provenance.producer_component == COLLABORATIVE_DECISION_BINDING_EVIDENCE_PRODUCER
    assert evidence.provenance.operation_id == COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID
    assert evidence.operation_outcome is not None
    assert evidence.operation_outcome.status is PipelineOperationStatus.SUCCEEDED


def test_tenant_mismatch_fail_closed() -> None:
    provenance = ExecutionProvenanceRef(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )
    correlation = execution_correlation_from_provenance(tenant_id="tenant-b", execution=provenance)
    with pytest.raises(ValueError, match="tenant_id"):
        append_decision_binding_create_outcome_evidence(
            _RecordingPersistence(),
            DefaultCollaborativeFunctionalEvidenceProjection(),
            binding=_binding(),
            operation_status=PipelineOperationStatus.SUCCEEDED,
            execution_correlation=correlation,
            recorded_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_duplicate_evidence_id_idempotent_replay() -> None:
    provenance = ExecutionProvenanceRef(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )
    correlation = execution_correlation_from_provenance(tenant_id="tenant-a", execution=provenance)
    evidence_id: EventId = mint_event_id()
    recorded_at = datetime(2026, 1, 2, tzinfo=UTC)
    persistence = _RecordingPersistence()
    strategy = DefaultCollaborativeFunctionalEvidenceProjection()
    first = append_decision_binding_create_outcome_evidence(
        persistence,
        strategy,
        binding=_binding(),
        operation_status=PipelineOperationStatus.SUCCEEDED,
        execution_correlation=correlation,
        recorded_at=recorded_at,
        evidence_id=evidence_id,
    )
    second = append_decision_binding_create_outcome_evidence(
        persistence,
        strategy,
        binding=_binding(),
        operation_status=PipelineOperationStatus.SUCCEEDED,
        execution_correlation=correlation,
        recorded_at=recorded_at,
        evidence_id=evidence_id,
    )
    assert first.evidence_id == second.evidence_id


def test_duplicate_evidence_id_different_content_conflicts() -> None:
    provenance = ExecutionProvenanceRef(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )
    correlation = execution_correlation_from_provenance(tenant_id="tenant-a", execution=provenance)
    evidence_id: EventId = mint_event_id()
    persistence = _RecordingPersistence()
    strategy = DefaultCollaborativeFunctionalEvidenceProjection()
    append_decision_binding_create_outcome_evidence(
        persistence,
        strategy,
        binding=_binding(),
        operation_status=PipelineOperationStatus.SUCCEEDED,
        execution_correlation=correlation,
        recorded_at=datetime(2026, 1, 2, tzinfo=UTC),
        evidence_id=evidence_id,
    )
    with pytest.raises(FunctionalEvidencePersistenceConflictError):
        append_decision_binding_create_outcome_evidence(
            persistence,
            strategy,
            binding=_binding(),
            operation_status=PipelineOperationStatus.FAILED,
            execution_correlation=correlation,
            recorded_at=datetime(2026, 1, 2, tzinfo=UTC),
            evidence_id=evidence_id,
        )
