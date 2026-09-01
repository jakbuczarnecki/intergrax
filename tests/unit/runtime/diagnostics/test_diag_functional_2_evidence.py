# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_event_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineArtifactLineageFact,
    PipelineCandidateFact,
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineOperationOutcomeFact,
    PipelineOperationStatus,
    PipelineOutputRelationFact,
    PipelineSelectionFact,
    PipelineValidationLinkFact,
    PlatformFunctionalEvidence,
    ScoreSemantics,
    TypedPipelineScore,
)
from intergrax.runtime.diagnostics.functional_validation import (
    DiagnosticExecutionCorrelation,
    FunctionalValidationEvidence,
    FunctionalValidationKind,
    FunctionalValidationOutcome,
    FunctionalValidatorRef,
    ExpectedActualRelation,
    functional_validation_evidence_id,
)
from intergrax.runtime.observability.export_attributes import ObservabilityArtifactReference

pytestmark = pytest.mark.unit


def _scope() -> PipelineEvidenceScope:
    return PipelineEvidenceScope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )


def _provenance(operation_id: str) -> PipelineEvidenceProvenance:
    return PipelineEvidenceProvenance(
        producer_component="diag.test",
        operation_id=operation_id,
    )


def _artifact(ref: str) -> ObservabilityArtifactReference:
    return ObservabilityArtifactReference(artifact_ref=ref)


def _evidence(
    *,
    kind: PipelineEvidenceKind,
    scope: PipelineEvidenceScope,
    provenance: PipelineEvidenceProvenance,
    **payload: object,
) -> PlatformFunctionalEvidence:
    return PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=kind,
        scope=scope,
        provenance=provenance,
        **payload,
    )


def test_e1_artifact_lineage() -> None:
    scope = _scope()
    evidence = _evidence(
        kind=PipelineEvidenceKind.ARTIFACT_LINEAGE,
        scope=scope,
        provenance=_provenance("derive-chunk"),
        artifact_lineage=PipelineArtifactLineageFact(
            source_artifact_ref=_artifact("doc:source-1"),
            derived_artifact_ref=_artifact("chunk:derived-1"),
            lineage_operation="chunk",
        ),
    )

    assert evidence.artifact_lineage is not None
    assert evidence.artifact_lineage.lineage_operation == "chunk"


def test_e2_operation_outcome() -> None:
    scope = _scope()
    evidence = _evidence(
        kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        scope=scope,
        provenance=_provenance("embed"),
        operation_outcome=PipelineOperationOutcomeFact(
            operation_name="embed",
            status=PipelineOperationStatus.SUCCEEDED,
            output_artifact_ref=_artifact("embedding:1"),
        ),
    )

    assert evidence.operation_outcome is not None
    assert evidence.operation_outcome.status is PipelineOperationStatus.SUCCEEDED


def test_e3_candidate_with_typed_rank_and_score() -> None:
    scope = _scope()
    evidence = _evidence(
        kind=PipelineEvidenceKind.CANDIDATE_RANK,
        scope=scope,
        provenance=_provenance("retrieve"),
        candidate=PipelineCandidateFact(
            query_id="query-1",
            candidate_artifact_ref=_artifact("candidate:17"),
            score=TypedPipelineScore(
                raw_value=0.73,
                semantics=ScoreSemantics.PROVIDER_OPAQUE,
                scale_hint="cosine_distance",
            ),
            rank=17,
            selected=False,
        ),
    )

    assert evidence.candidate is not None
    assert evidence.candidate.rank == 17
    assert evidence.candidate.score is not None
    assert evidence.candidate.score.semantics is ScoreSemantics.PROVIDER_OPAQUE


def test_e4_selection() -> None:
    scope = _scope()
    evidence = _evidence(
        kind=PipelineEvidenceKind.SELECTION,
        scope=scope,
        provenance=_provenance("rank-select"),
        selection=PipelineSelectionFact(
            query_id="query-1",
            selected_artifact_ref=_artifact("candidate:3"),
            candidate_count=12,
            selection_reason="top_rank",
        ),
    )

    assert evidence.selection is not None
    assert evidence.selection.candidate_count == 12


def test_e5_output_relation() -> None:
    scope = _scope()
    evidence = _evidence(
        kind=PipelineEvidenceKind.OUTPUT_RELATION,
        scope=scope,
        provenance=_provenance("synthesize"),
        output_relation=PipelineOutputRelationFact(
            selected_artifact_ref=_artifact("candidate:3"),
            output_artifact_ref=_artifact("answer:1"),
            relation_kind="generated_from",
        ),
    )

    assert evidence.output_relation is not None


def test_e6_functional_validation_link() -> None:
    scope = _scope()
    correlation = DiagnosticExecutionCorrelation(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    validation_id = functional_validation_evidence_id(
        validator_id="oracle.v1",
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        correlation=correlation,
        idempotency_key="attempt-1",
    )
    validation = FunctionalValidationEvidence(
        validation_id=validation_id,
        validator=FunctionalValidatorRef(validator_id="oracle.v1"),
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        outcome=FunctionalValidationOutcome.FAILED,
        correlation=correlation,
        expected_actual_relation=ExpectedActualRelation.CONTAINS,
    )
    evidence = _evidence(
        kind=PipelineEvidenceKind.VALIDATION,
        scope=scope,
        provenance=_provenance("validate"),
        validation_link=PipelineValidationLinkFact(
            validation_id=validation.validation_id,
            output_artifact_ref=_artifact("answer:1"),
        ),
    )

    assert evidence.validation_link is not None
    assert evidence.validation_link.validation_id == validation.validation_id


def test_kind_payload_mismatch_is_rejected() -> None:
    scope = _scope()
    with pytest.raises(ValueError):
        PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.CANDIDATE_RANK,
            scope=scope,
            provenance=_provenance("retrieve"),
            operation_outcome=PipelineOperationOutcomeFact(
                operation_name="retrieve",
                status=PipelineOperationStatus.SUCCEEDED,
            ),
        )
