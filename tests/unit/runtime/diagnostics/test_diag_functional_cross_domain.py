# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_event_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.functional_evidence import (
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
    ExpectedActualRelation,
    FunctionalValidationEvidence,
    FunctionalValidationKind,
    FunctionalValidationOutcome,
    FunctionalValidatorRef,
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


def _artifact(ref: str) -> ObservabilityArtifactReference:
    return ObservabilityArtifactReference(artifact_ref=ref)


def test_retrieval_candidate_selection_uses_generic_candidate_contract() -> None:
    scope = _scope()
    candidate = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.CANDIDATE_RANK,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="pipeline.retrieval",
            operation_id="retrieve-rank",
        ),
        candidate=PipelineCandidateFact(
            query_id="retrieval-query-1",
            candidate_artifact_ref=_artifact("context:chunk-9"),
            score=TypedPipelineScore(
                raw_value=0.42,
                semantics=ScoreSemantics.HIGHER_IS_BETTER,
            ),
            rank=2,
            selected=True,
        ),
    )
    selection = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.SELECTION,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="pipeline.retrieval",
            operation_id="select-context",
        ),
        selection=PipelineSelectionFact(
            query_id="retrieval-query-1",
            selected_artifact_ref=_artifact("context:chunk-9"),
            candidate_count=8,
            selection_reason="top_rank",
        ),
    )

    assert candidate.kind is PipelineEvidenceKind.CANDIDATE_RANK
    assert selection.kind is PipelineEvidenceKind.SELECTION
    assert candidate.candidate is not None
    assert selection.selection is not None


def test_tool_candidate_selection_uses_same_generic_contract() -> None:
    scope = _scope()
    candidate = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.CANDIDATE_RANK,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="pipeline.tool_router",
            operation_id="rank-tools",
        ),
        candidate=PipelineCandidateFact(
            query_id="tool-routing-1",
            candidate_artifact_ref=_artifact("tool:search_web"),
            score=TypedPipelineScore(
                raw_value=0.91,
                semantics=ScoreSemantics.PROBABILITY,
            ),
            rank=1,
            selected=True,
        ),
    )
    selection = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.SELECTION,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="pipeline.tool_router",
            operation_id="select-tool",
        ),
        selection=PipelineSelectionFact(
            query_id="tool-routing-1",
            selected_artifact_ref=_artifact("tool:search_web"),
            candidate_count=5,
            selection_reason="policy_rank_1",
        ),
    )

    assert candidate.candidate is not None
    assert selection.selection is not None
    assert candidate.candidate.query_id == selection.selection.query_id


def test_model_routing_candidate_selection_uses_same_generic_contract() -> None:
    scope = _scope()
    candidate = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.CANDIDATE_RANK,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="pipeline.model_router",
            operation_id="rank-models",
        ),
        candidate=PipelineCandidateFact(
            query_id="model-route-1",
            candidate_artifact_ref=_artifact("model:small-fast"),
            score=TypedPipelineScore(
                raw_value=0.67,
                semantics=ScoreSemantics.LOWER_IS_BETTER,
                scale_hint="latency_ms",
            ),
            rank=1,
            selected=False,
        ),
    )

    assert candidate.candidate is not None
    assert candidate.candidate.candidate_artifact_ref.artifact_ref == "model:small-fast"


def test_web_search_candidate_selection_output_validation_uses_generic_contract() -> None:
    scope = _scope()
    operation = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="pipeline.web_search",
            operation_id="execute-search",
        ),
        operation_outcome=PipelineOperationOutcomeFact(
            operation_name="web_search",
            status=PipelineOperationStatus.SUCCEEDED,
        ),
    )
    candidate = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.CANDIDATE_RANK,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="pipeline.web_search",
            operation_id="rank-results",
        ),
        candidate=PipelineCandidateFact(
            query_id="web-search-1",
            candidate_artifact_ref=_artifact("web:result-2"),
            score=TypedPipelineScore(
                raw_value=0.88,
                semantics=ScoreSemantics.HIGHER_IS_BETTER,
            ),
            rank=1,
            selected=True,
        ),
    )
    selection = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.SELECTION,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="pipeline.web_search",
            operation_id="select-source",
        ),
        selection=PipelineSelectionFact(
            query_id="web-search-1",
            selected_artifact_ref=_artifact("web:result-2"),
            candidate_count=6,
            selection_reason="top_rank",
        ),
    )
    output = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.OUTPUT_RELATION,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="pipeline.web_search",
            operation_id="synthesize-answer",
        ),
        output_relation=PipelineOutputRelationFact(
            selected_artifact_ref=_artifact("web:result-2"),
            output_artifact_ref=_artifact("answer:web-1"),
            relation_kind="cited_from",
        ),
    )
    correlation = DiagnosticExecutionCorrelation(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    validation_id = functional_validation_evidence_id(
        validator_id="web.oracle.v1",
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        correlation=correlation,
        idempotency_key="web-search-attempt-1",
    )
    validation = FunctionalValidationEvidence(
        validation_id=validation_id,
        validator=FunctionalValidatorRef(validator_id="web.oracle.v1"),
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        outcome=FunctionalValidationOutcome.PASSED,
        correlation=correlation,
        expected_actual_relation=ExpectedActualRelation.CONTAINS,
    )
    validation_link = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.VALIDATION,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="pipeline.web_search",
            operation_id="validate-answer",
        ),
        validation_link=PipelineValidationLinkFact(
            validation_id=validation.validation_id,
            output_artifact_ref=_artifact("answer:web-1"),
        ),
    )

    assert operation.operation_outcome is not None
    assert candidate.candidate is not None
    assert selection.selection is not None
    assert output.output_relation is not None
    assert validation_link.validation_link is not None
    assert selection.selection.selected_artifact_ref == candidate.candidate.candidate_artifact_ref
    assert output.output_relation.selected_artifact_ref == selection.selection.selected_artifact_ref
