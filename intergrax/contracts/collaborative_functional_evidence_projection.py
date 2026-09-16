# © Artur Czarnecki. All rights reserved.

"""Multiplayer → canonical Evidence Plane projection contracts (MP-4R5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from intergrax.contracts.collaborative_decision_binding import CollaborativeDecisionBinding
from intergrax.contracts.execution_identity import EventId
from intergrax.contracts.functional_evidence.correlation import (
    FunctionalEvidenceExecutionCorrelation,
)
from intergrax.contracts.functional_evidence.models import (
    PipelineOperationStatus,
    PlatformFunctionalEvidence,
)

COLLABORATIVE_DECISION_BINDING_EVIDENCE_PRODUCER: Final = "collaborative_work.decision_binding"
COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID: Final = (
    "collaborative_work.decision_binding.create"
)


class CollaborativeFunctionalEvidenceProjectionError(Exception):
    """Base error for collaborative functional evidence projection."""


class CollaborativeFunctionalEvidenceNotApplicable(CollaborativeFunctionalEvidenceProjectionError):
    """Raised when canonical functional evidence cannot be emitted for the input."""


class CollaborativeDecisionBindingAssociationNotRepresentable(
    CollaborativeFunctionalEvidenceProjectionError,
):
    """
    Collaborative Work ↔ DecisionProposal association has no semantic fit in frozen Evidence Plane.

    Requires platform Evidence Plane contract extension (architecture decision) — not MP-4R5 local workaround.
    """


@dataclass(frozen=True, slots=True)
class CollaborativeDecisionBindingCreateOutcomeProjection:
    """Inputs for projecting a decision-binding create operation outcome fact only."""

    binding: CollaborativeDecisionBinding
    operation_status: PipelineOperationStatus
    execution_correlation: FunctionalEvidenceExecutionCorrelation
    recorded_at: datetime
    evidence_id: EventId | None = None


class CollaborativeFunctionalEvidenceProjectionStrategy(ABC):
    """
    Maps Multiplayer-owned facts to already-approved ``PlatformFunctionalEvidence`` facts.

    Strategies may choose which applicable canonical facts to emit; they must not define new
    evidence authority or reinterpret frozen ``PipelineEvidenceKind`` semantics.
    """

    @abstractmethod
    def project_decision_binding_create_outcome(
        self,
        projection: CollaborativeDecisionBindingCreateOutcomeProjection,
    ) -> PlatformFunctionalEvidence:
        """Emit ``OPERATION_OUTCOME`` for the binding create operation when execution correlation is known."""

    @abstractmethod
    def project_decision_binding_association(
        self,
        binding: CollaborativeDecisionBinding,
        *,
        execution_correlation: FunctionalEvidenceExecutionCorrelation | None,
    ) -> tuple[PlatformFunctionalEvidence, ...]:
        """
        Attempt to project the immutable binding association into canonical functional evidence.

        Frozen Evidence Plane has no matching kind — default implementation raises
        ``CollaborativeDecisionBindingAssociationNotRepresentable``.
        """


__all__ = [
    "COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID",
    "COLLABORATIVE_DECISION_BINDING_EVIDENCE_PRODUCER",
    "CollaborativeDecisionBindingAssociationNotRepresentable",
    "CollaborativeDecisionBindingCreateOutcomeProjection",
    "CollaborativeFunctionalEvidenceNotApplicable",
    "CollaborativeFunctionalEvidenceProjectionError",
    "CollaborativeFunctionalEvidenceProjectionStrategy",
]
