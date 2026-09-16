# © Artur Czarnecki. All rights reserved.

"""Composition-facing decision binding evidence adoption (MP-4R5)."""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.collaborative_decision_binding import CollaborativeDecisionBinding
from intergrax.contracts.collaborative_functional_evidence_projection import (
    CollaborativeDecisionBindingCreateOutcomeProjection,
    CollaborativeFunctionalEvidenceNotApplicable,
    CollaborativeFunctionalEvidenceProjectionStrategy,
)
from intergrax.contracts.execution_identity import EventId
from intergrax.contracts.functional_evidence.correlation import (
    FunctionalEvidenceExecutionCorrelation,
)
from intergrax.contracts.functional_evidence.models import (
    PipelineOperationStatus,
    PlatformFunctionalEvidence,
)
from intergrax.contracts.functional_evidence.persistence import FunctionalEvidencePersistence


def append_decision_binding_create_outcome_evidence(
    persistence: FunctionalEvidencePersistence,
    strategy: CollaborativeFunctionalEvidenceProjectionStrategy,
    *,
    binding: CollaborativeDecisionBinding,
    operation_status: PipelineOperationStatus,
    execution_correlation: FunctionalEvidenceExecutionCorrelation | None,
    recorded_at: datetime,
    evidence_id: EventId | None = None,
) -> PlatformFunctionalEvidence:
    """
    Append canonical operation-outcome evidence for a successful/failed binding create.

    Association truth remains in ``CollaborativeDecisionBindingRepository``; this records
    operation execution only. Without execution correlation, raises ``CollaborativeFunctionalEvidenceNotApplicable``.
    """
    if execution_correlation is None:
        raise CollaborativeFunctionalEvidenceNotApplicable(
            "decision binding create evidence requires canonical execution correlation",
        )
    evidence = strategy.project_decision_binding_create_outcome(
        CollaborativeDecisionBindingCreateOutcomeProjection(
            binding=binding,
            operation_status=operation_status,
            execution_correlation=execution_correlation,
            recorded_at=recorded_at,
            evidence_id=evidence_id,
        ),
    )
    return persistence.append(evidence)


__all__ = ["append_decision_binding_create_outcome_evidence"]
