# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_event_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineCandidateFact,
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineSelectionFact,
    PlatformFunctionalEvidence,
    ScoreSemantics,
    TypedPipelineScore,
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
