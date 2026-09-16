# © Artur Czarnecki. All rights reserved.

"""Default Multiplayer functional evidence projection (MP-4R5)."""

from __future__ import annotations

from intergrax.contracts.collaborative_decision_binding import CollaborativeDecisionBinding
from intergrax.contracts.collaborative_functional_evidence_projection import (
    COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID,
    COLLABORATIVE_DECISION_BINDING_EVIDENCE_PRODUCER,
    CollaborativeDecisionBindingAssociationNotRepresentable,
    CollaborativeDecisionBindingCreateOutcomeProjection,
    CollaborativeFunctionalEvidenceProjectionStrategy,
)
from intergrax.contracts.execution_identity import EventId, mint_event_id
from intergrax.contracts.functional_evidence.correlation import (
    FunctionalEvidenceExecutionCorrelation,
)
from intergrax.contracts.functional_evidence.models import (
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineOperationOutcomeFact,
    PlatformFunctionalEvidence,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef

_COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_NAME = COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID


def execution_correlation_from_provenance(
    *,
    tenant_id: str,
    execution: ExecutionProvenanceRef,
) -> FunctionalEvidenceExecutionCorrelation:
    """Build five-ID functional evidence correlation from canonical execution provenance + tenant."""
    normalized_tenant = tenant_id.strip()
    if not normalized_tenant or tenant_id != normalized_tenant:
        raise ValueError("tenant_id must be non-empty and normalized")
    return FunctionalEvidenceExecutionCorrelation(
        tenant_id=normalized_tenant,
        task_id=execution.task_id,
        run_id=execution.run_id,
        attempt_id=execution.attempt_id,
        execution_id=execution.execution_id,
    )


class DefaultCollaborativeFunctionalEvidenceProjection(
    CollaborativeFunctionalEvidenceProjectionStrategy,
):
    """Maps Multiplayer facts to frozen canonical functional evidence kinds only."""

    def project_decision_binding_create_outcome(
        self,
        projection: CollaborativeDecisionBindingCreateOutcomeProjection,
    ) -> PlatformFunctionalEvidence:
        binding = projection.binding
        if binding.tenant_id != projection.execution_correlation.tenant_id:
            raise ValueError("execution correlation tenant_id must match binding tenant_id")
        scope = PipelineEvidenceScope.from_correlation(projection.execution_correlation)
        evidence_id = projection.evidence_id if projection.evidence_id is not None else mint_event_id()
        return PlatformFunctionalEvidence(
            evidence_id=evidence_id,
            kind=PipelineEvidenceKind.OPERATION_OUTCOME,
            scope=scope,
            provenance=PipelineEvidenceProvenance(
                producer_component=COLLABORATIVE_DECISION_BINDING_EVIDENCE_PRODUCER,
                operation_id=COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID,
                recorded_at=projection.recorded_at,
            ),
            operation_outcome=PipelineOperationOutcomeFact(
                operation_name=_COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_NAME,
                status=projection.operation_status,
            ),
        )

    def project_decision_binding_association(
        self,
        binding: CollaborativeDecisionBinding,
        *,
        execution_correlation: FunctionalEvidenceExecutionCorrelation | None,
    ) -> tuple[PlatformFunctionalEvidence, ...]:
        raise CollaborativeDecisionBindingAssociationNotRepresentable(
            "CollaborativeDecisionBinding association is Multiplayer source-of-truth only; "
            "frozen PipelineEvidenceKind set has no exact semantic fit for WorkItem ↔ DecisionProposalRef",
        )


__all__ = [
    "DefaultCollaborativeFunctionalEvidenceProjection",
    "execution_correlation_from_provenance",
]
