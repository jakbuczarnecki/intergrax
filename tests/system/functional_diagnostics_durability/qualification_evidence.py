# © Artur Czarnecki. All rights reserved.

"""Cross-domain functional evidence fixtures for D1-R1 qualification."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from intergrax.contracts.execution_identity import mint_event_id
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
from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
  C1_RAG_EXPECTED_SELECTION_ARTIFACT,
  C1_RAG_QUERY_ID,
  C1_RAG_RETRIEVE_OPERATION_ID,
)
from intergrax.runtime.diagnostics.specifications.q2_tool_selection_functional_diagnostic_specification import (
  Q2_EXPECTED_SEARCH_TOOL_ARTIFACT,
  Q2_TOOL_INVOKE_OPERATION_ID,
  Q2_TOOL_QUERY_ID,
)
from intergrax.runtime.diagnostics.specifications.q3_web_search_functional_diagnostic_specification import (
  Q3_WEB_QUERY_ID,
  Q3_WEB_SEARCH_OPERATION_ID,
)
from intergrax.runtime.diagnostics.specifications.q4_model_routing_functional_diagnostic_specification import (
  Q4_MODEL_GENERATE_OPERATION_ID,
  Q4_MODEL_QUERY_ID,
)
from intergrax.runtime.observability.export_attributes import ObservabilityArtifactReference

_BASE_TIME = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def build_assessment_recovery_evidence(
  scope: PipelineEvidenceScope,
) -> tuple[PlatformFunctionalEvidence, ...]:
  """Evidence sufficient for meaningful C1 RAG diagnostic assessment."""
  return (
    PlatformFunctionalEvidence(
      evidence_id=mint_event_id(),
      kind=PipelineEvidenceKind.OPERATION_OUTCOME,
      scope=scope,
      provenance=PipelineEvidenceProvenance(
        producer_component="domain.rag",
        operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
        recorded_at=_BASE_TIME,
      ),
      operation_outcome=PipelineOperationOutcomeFact(
        operation_name="retrieve",
        status=PipelineOperationStatus.SUCCEEDED,
      ),
    ),
    PlatformFunctionalEvidence(
      evidence_id=mint_event_id(),
      kind=PipelineEvidenceKind.CANDIDATE_RANK,
      scope=scope,
      provenance=PipelineEvidenceProvenance(
        producer_component="domain.rag",
        operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
        recorded_at=_BASE_TIME + timedelta(seconds=1),
      ),
      candidate=PipelineCandidateFact(
        query_id=C1_RAG_QUERY_ID,
        candidate_artifact_ref=ObservabilityArtifactReference(
          artifact_ref="chunk:candidate",
        ),
        rank=1,
        selected=True,
      ),
    ),
    PlatformFunctionalEvidence(
      evidence_id=mint_event_id(),
      kind=PipelineEvidenceKind.SELECTION,
      scope=scope,
      provenance=PipelineEvidenceProvenance(
        producer_component="domain.rag",
        operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
        recorded_at=_BASE_TIME + timedelta(seconds=2),
      ),
      selection=PipelineSelectionFact(
        query_id=C1_RAG_QUERY_ID,
        selected_artifact_ref=ObservabilityArtifactReference(
          artifact_ref=C1_RAG_EXPECTED_SELECTION_ARTIFACT,
        ),
        candidate_count=2,
        selection_reason="top_score",
      ),
    ),
  )


def build_cross_domain_codec_evidence(
  scope: PipelineEvidenceScope,
) -> tuple[PlatformFunctionalEvidence, ...]:
  """All six evidence kinds with domain-tagged producer components."""
  assessment = build_assessment_recovery_evidence(scope)
  tools_evidence = PlatformFunctionalEvidence(
    evidence_id=mint_event_id(),
    kind=PipelineEvidenceKind.OPERATION_OUTCOME,
    scope=scope,
    provenance=PipelineEvidenceProvenance(
      producer_component="domain.tools",
      operation_id=Q2_TOOL_INVOKE_OPERATION_ID,
      recorded_at=_BASE_TIME + timedelta(seconds=3),
    ),
    operation_outcome=PipelineOperationOutcomeFact(
      operation_name="tool.invoke",
      status=PipelineOperationStatus.SUCCEEDED,
    ),
  )
  web_evidence = PlatformFunctionalEvidence(
    evidence_id=mint_event_id(),
    kind=PipelineEvidenceKind.CANDIDATE_RANK,
    scope=scope,
    provenance=PipelineEvidenceProvenance(
      producer_component="domain.web",
      operation_id=Q3_WEB_SEARCH_OPERATION_ID,
      recorded_at=_BASE_TIME + timedelta(seconds=4),
    ),
    candidate=PipelineCandidateFact(
      query_id=Q3_WEB_QUERY_ID,
      candidate_artifact_ref=ObservabilityArtifactReference(artifact_ref="web:result"),
      score=TypedPipelineScore(raw_value=0.8, semantics=ScoreSemantics.HIGHER_IS_BETTER),
      rank=1,
      selected=True,
    ),
  )
  model_evidence = PlatformFunctionalEvidence(
    evidence_id=mint_event_id(),
    kind=PipelineEvidenceKind.SELECTION,
    scope=scope,
    provenance=PipelineEvidenceProvenance(
      producer_component="domain.model_routing",
      operation_id=Q4_MODEL_GENERATE_OPERATION_ID,
      recorded_at=_BASE_TIME + timedelta(seconds=5),
    ),
    selection=PipelineSelectionFact(
      query_id=Q4_MODEL_QUERY_ID,
      selected_artifact_ref=ObservabilityArtifactReference(artifact_ref="model:gpt-4"),
      candidate_count=2,
      selection_reason="routing_policy",
    ),
  )
  lineage_evidence = PlatformFunctionalEvidence(
    evidence_id=mint_event_id(),
    kind=PipelineEvidenceKind.ARTIFACT_LINEAGE,
    scope=scope,
    provenance=PipelineEvidenceProvenance(
      producer_component="domain.tools",
      operation_id=Q2_TOOL_INVOKE_OPERATION_ID,
      recorded_at=_BASE_TIME + timedelta(seconds=6),
    ),
    artifact_lineage=PipelineArtifactLineageFact(
      source_artifact_ref=ObservabilityArtifactReference(artifact_ref=Q2_EXPECTED_SEARCH_TOOL_ARTIFACT),
      derived_artifact_ref=ObservabilityArtifactReference(artifact_ref="tool:result"),
      lineage_operation="invoke",
    ),
  )
  output_relation_evidence = PlatformFunctionalEvidence(
    evidence_id=mint_event_id(),
    kind=PipelineEvidenceKind.OUTPUT_RELATION,
    scope=scope,
    provenance=PipelineEvidenceProvenance(
      producer_component="domain.model_routing",
      operation_id=Q4_MODEL_GENERATE_OPERATION_ID,
      recorded_at=_BASE_TIME + timedelta(seconds=7),
    ),
    output_relation=PipelineOutputRelationFact(
      selected_artifact_ref=ObservabilityArtifactReference(artifact_ref="model:gpt-4"),
      output_artifact_ref=ObservabilityArtifactReference(artifact_ref="output:generated"),
      relation_kind="derived_from",
    ),
  )
  validation_evidence = PlatformFunctionalEvidence(
    evidence_id=mint_event_id(),
    kind=PipelineEvidenceKind.VALIDATION,
    scope=scope,
    provenance=PipelineEvidenceProvenance(
      producer_component="domain.web",
      operation_id=Q3_WEB_SEARCH_OPERATION_ID,
      recorded_at=_BASE_TIME + timedelta(seconds=8),
    ),
    validation_link=PipelineValidationLinkFact(
      validation_id=mint_event_id(),
      output_artifact_ref=ObservabilityArtifactReference(artifact_ref="output:generated"),
    ),
  )
  return (
    *assessment,
    tools_evidence,
    web_evidence,
    model_evidence,
    lineage_evidence,
    output_relation_evidence,
    validation_evidence,
  )


def build_pagination_evidence(
  scope: PipelineEvidenceScope,
) -> tuple[PlatformFunctionalEvidence, ...]:
  return tuple(
    PlatformFunctionalEvidence(
      evidence_id=mint_event_id(),
      kind=PipelineEvidenceKind.OPERATION_OUTCOME,
      scope=scope,
      provenance=PipelineEvidenceProvenance(
        producer_component="domain.rag",
        operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
        recorded_at=_BASE_TIME + timedelta(seconds=index),
      ),
      operation_outcome=PipelineOperationOutcomeFact(
        operation_name=f"pagination-op-{index}",
        status=PipelineOperationStatus.SUCCEEDED,
      ),
    )
    for index in range(4)
  )


__all__ = [
  "build_assessment_recovery_evidence",
  "build_cross_domain_codec_evidence",
  "build_pagination_evidence",
]
