# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Append-only functional evidence recorder for real execution paths (DIAG-FUNCTIONAL-Q1)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.execution_identity import EventId, mint_event_id
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
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
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistence,
)
from intergrax.runtime.observability.export_attributes import ObservabilityArtifactReference

_FUNCTIONAL_EVIDENCE_RECORDER_KEY = "functional_evidence_recorder"
_SUPPRESS_KINDS_METADATA_KEY = "qualification_suppress_functional_evidence_kinds"
_PRODUCER_COMPONENT = "agents.local_search"


def functional_evidence_recorder_key() -> str:
    return _FUNCTIONAL_EVIDENCE_RECORDER_KEY


def suppress_kinds_from_metadata(metadata: dict[str, Any]) -> frozenset[PipelineEvidenceKind]:
    raw = metadata.get(_SUPPRESS_KINDS_METADATA_KEY)
    if not isinstance(raw, list):
        return frozenset()
    kinds: set[PipelineEvidenceKind] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        try:
            kinds.add(PipelineEvidenceKind(item.strip()))
        except ValueError:
            continue
    return frozenset(kinds)


def attach_functional_evidence_recorder(
    exec_ctx: RuntimeExecutionContext,
    recorder: FunctionalEvidenceRecorder,
) -> None:
    exec_ctx.metadata[_FUNCTIONAL_EVIDENCE_RECORDER_KEY] = recorder


def recorder_from_exec_ctx(exec_ctx: RuntimeExecutionContext | None) -> FunctionalEvidenceRecorder | None:
    if exec_ctx is None:
        return None
    raw = exec_ctx.metadata.get(_FUNCTIONAL_EVIDENCE_RECORDER_KEY)
    if isinstance(raw, FunctionalEvidenceRecorder):
        return raw
    return None


def artifact_ref(value: str) -> ObservabilityArtifactReference:
    return ObservabilityArtifactReference(artifact_ref=value)


class FunctionalEvidenceRecorder:
    """Records typed pipeline facts only — no diagnosis."""

    def __init__(
        self,
        persistence: FunctionalEvidencePersistence,
        *,
        producer_component: str = _PRODUCER_COMPONENT,
    ) -> None:
        self._persistence = persistence
        self._producer_component = producer_component

    @property
    def persistence(self) -> FunctionalEvidencePersistence:
        return self._persistence

    def scope_from_exec_ctx(
        self,
        exec_ctx: RuntimeExecutionContext,
        *,
        tenant_id: str | None = None,
    ) -> PipelineEvidenceScope:
        resolved_tenant = tenant_id
        if resolved_tenant is None and exec_ctx.request is not None:
            request_tenant = getattr(exec_ctx.request, "tenant_id", None)
            if request_tenant and str(request_tenant).strip():
                resolved_tenant = str(request_tenant).strip()
        if resolved_tenant is None:
            resolved_tenant = "default"
        return PipelineEvidenceScope(
            tenant_id=resolved_tenant,
            task_id=exec_ctx.task_id,
            run_id=exec_ctx.run_id,
            attempt_id=exec_ctx.attempt_id,
        )

    def record_operation_outcome(
        self,
        *,
        scope: PipelineEvidenceScope,
        operation_id: str,
        operation_name: str,
        status: PipelineOperationStatus,
        suppressed_kinds: frozenset[PipelineEvidenceKind] | None = None,
    ) -> PlatformFunctionalEvidence | None:
        if self._is_suppressed(PipelineEvidenceKind.OPERATION_OUTCOME, suppressed_kinds):
            return None
        evidence = PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.OPERATION_OUTCOME,
            scope=scope,
            provenance=PipelineEvidenceProvenance(
                producer_component=self._producer_component,
                operation_id=operation_id,
            ),
            operation_outcome=PipelineOperationOutcomeFact(
                operation_name=operation_name,
                status=status,
            ),
        )
        return self._persistence.append(evidence)

    def record_candidate_rank(
        self,
        *,
        scope: PipelineEvidenceScope,
        operation_id: str,
        query_id: str,
        candidate_artifact_ref: str,
        rank: int,
        selected: bool,
        score: float | None = None,
        suppressed_kinds: frozenset[PipelineEvidenceKind] | None = None,
    ) -> PlatformFunctionalEvidence | None:
        if self._is_suppressed(PipelineEvidenceKind.CANDIDATE_RANK, suppressed_kinds):
            return None
        typed_score = None
        if score is not None:
            typed_score = TypedPipelineScore(
                raw_value=float(score),
                semantics=ScoreSemantics.HIGHER_IS_BETTER,
            )
        evidence = PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.CANDIDATE_RANK,
            scope=scope,
            provenance=PipelineEvidenceProvenance(
                producer_component=self._producer_component,
                operation_id=operation_id,
            ),
            candidate=PipelineCandidateFact(
                query_id=query_id,
                candidate_artifact_ref=artifact_ref(candidate_artifact_ref),
                score=typed_score,
                rank=rank,
                selected=selected,
            ),
        )
        return self._persistence.append(evidence)

    def record_selection(
        self,
        *,
        scope: PipelineEvidenceScope,
        operation_id: str,
        query_id: str,
        selected_artifact_ref: str,
        candidate_count: int,
        selection_reason: str = "",
        suppressed_kinds: frozenset[PipelineEvidenceKind] | None = None,
    ) -> PlatformFunctionalEvidence | None:
        if self._is_suppressed(PipelineEvidenceKind.SELECTION, suppressed_kinds):
            return None
        evidence = PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.SELECTION,
            scope=scope,
            provenance=PipelineEvidenceProvenance(
                producer_component=self._producer_component,
                operation_id=operation_id,
            ),
            selection=PipelineSelectionFact(
                query_id=query_id,
                selected_artifact_ref=artifact_ref(selected_artifact_ref),
                candidate_count=candidate_count,
                selection_reason=selection_reason,
            ),
        )
        return self._persistence.append(evidence)

    def record_output_relation(
        self,
        *,
        scope: PipelineEvidenceScope,
        operation_id: str,
        selected_artifact_ref: str,
        output_artifact_ref: str,
        relation_kind: str,
        suppressed_kinds: frozenset[PipelineEvidenceKind] | None = None,
    ) -> PlatformFunctionalEvidence | None:
        if self._is_suppressed(PipelineEvidenceKind.OUTPUT_RELATION, suppressed_kinds):
            return None
        evidence = PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.OUTPUT_RELATION,
            scope=scope,
            provenance=PipelineEvidenceProvenance(
                producer_component=self._producer_component,
                operation_id=operation_id,
            ),
            output_relation=PipelineOutputRelationFact(
                selected_artifact_ref=artifact_ref(selected_artifact_ref),
                output_artifact_ref=artifact_ref(output_artifact_ref),
                relation_kind=relation_kind,
            ),
        )
        return self._persistence.append(evidence)

    def record_validation_link(
        self,
        *,
        scope: PipelineEvidenceScope,
        operation_id: str,
        validation_id: EventId,
        output_artifact_ref: str | None = None,
        suppressed_kinds: frozenset[PipelineEvidenceKind] | None = None,
    ) -> PlatformFunctionalEvidence | None:
        if self._is_suppressed(PipelineEvidenceKind.VALIDATION, suppressed_kinds):
            return None
        evidence = PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.VALIDATION,
            scope=scope,
            provenance=PipelineEvidenceProvenance(
                producer_component=self._producer_component,
                operation_id=operation_id,
            ),
            validation_link=PipelineValidationLinkFact(
                validation_id=validation_id,
                output_artifact_ref=artifact_ref(output_artifact_ref) if output_artifact_ref else None,
            ),
        )
        return self._persistence.append(evidence)

    @staticmethod
    def _is_suppressed(
        kind: PipelineEvidenceKind,
        suppressed_kinds: frozenset[PipelineEvidenceKind] | None,
    ) -> bool:
        return suppressed_kinds is not None and kind in suppressed_kinds


__all__ = [
    "FunctionalEvidenceRecorder",
    "artifact_ref",
    "attach_functional_evidence_recorder",
    "functional_evidence_recorder_key",
    "recorder_from_exec_ctx",
    "suppress_kinds_from_metadata",
]
