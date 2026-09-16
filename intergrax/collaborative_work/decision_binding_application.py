# © Artur Czarnecki. All rights reserved.

"""Application orchestration for Collaborative Decision Binding create + optional evidence (MP-4R5)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.collaborative_work.decision_binding_evidence import (
    append_decision_binding_create_outcome_evidence,
)
from intergrax.collaborative_work.decision_binding_service import CollaborativeDecisionBindingService
from intergrax.contracts.collaborative_decision_binding import (
    CollaborativeDecisionBinding,
    CreateCollaborativeDecisionBindingRequest,
)
from intergrax.contracts.collaborative_functional_evidence_projection import (
    CollaborativeFunctionalEvidenceProjectionStrategy,
)
from intergrax.contracts.execution_identity import EventId
from intergrax.contracts.functional_evidence.correlation import (
    FunctionalEvidenceExecutionCorrelation,
)
from intergrax.contracts.functional_evidence.models import PipelineOperationStatus
from intergrax.contracts.functional_evidence.persistence import FunctionalEvidencePersistence

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CollaborativeDecisionBindingEvidenceAdoption:
    """Explicitly configured evidence adoption for binding create (composition-selected)."""

    persistence: FunctionalEvidencePersistence
    strategy: CollaborativeFunctionalEvidenceProjectionStrategy


class CollaborativeDecisionBindingApplicationService:
    """
    Coordinates authoritative binding create with optional canonical operation-outcome evidence.

    Domain authority remains in ``CollaborativeDecisionBindingService``. Successful binding
    commits are not rolled back when evidence append fails; evidence exceptions propagate per
    ``FunctionalEvidencePersistence`` contract semantics. When binding create fails, failed
    operation-outcome evidence is best-effort (hosted bootstrap failure reporter semantics):
    secondary emission errors are logged and must not replace the primary domain failure.
    """

    def __init__(
        self,
        *,
        binding_service: CollaborativeDecisionBindingService,
        evidence_adoption: CollaborativeDecisionBindingEvidenceAdoption | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._binding_service = binding_service
        self._evidence_adoption = evidence_adoption
        self._clock = clock or (lambda: datetime.now(UTC))

    def create_binding(
        self,
        request: CreateCollaborativeDecisionBindingRequest,
        *,
        execution_correlation: FunctionalEvidenceExecutionCorrelation | None = None,
        evidence_id: EventId | None = None,
    ) -> CollaborativeDecisionBinding:
        recorded_at = self._require_timezone_aware(self._clock())
        adoption = self._evidence_adoption
        try:
            binding = self._binding_service.create_binding(request)
        except Exception:
            if adoption is not None and execution_correlation is not None:
                self._emit_failed_create_outcome_evidence_after_primary_failure(
                    adoption=adoption,
                    request=request,
                    execution_correlation=execution_correlation,
                    recorded_at=recorded_at,
                    evidence_id=evidence_id,
                )
            raise

        if adoption is not None and execution_correlation is not None:
            append_decision_binding_create_outcome_evidence(
                adoption.persistence,
                adoption.strategy,
                tenant_id=request.tenant_id,
                operation_status=PipelineOperationStatus.SUCCEEDED,
                execution_correlation=execution_correlation,
                recorded_at=recorded_at,
                evidence_id=evidence_id,
                binding=binding,
            )
        return binding

    @staticmethod
    def _emit_failed_create_outcome_evidence_after_primary_failure(
        *,
        adoption: CollaborativeDecisionBindingEvidenceAdoption,
        request: CreateCollaborativeDecisionBindingRequest,
        execution_correlation: FunctionalEvidenceExecutionCorrelation,
        recorded_at: datetime,
        evidence_id: EventId | None,
    ) -> None:
        try:
            append_decision_binding_create_outcome_evidence(
                adoption.persistence,
                adoption.strategy,
                tenant_id=request.tenant_id,
                operation_status=PipelineOperationStatus.FAILED,
                execution_correlation=execution_correlation,
                recorded_at=recorded_at,
                evidence_id=evidence_id,
                binding=None,
            )
        except Exception:
            _LOGGER.exception(
                "failed-operation-outcome evidence emission after binding create failure; "
                "primary operation failure is preserved",
            )

    @staticmethod
    def _require_timezone_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("clock must return timezone-aware datetime")
        return value


__all__ = [
    "CollaborativeDecisionBindingApplicationService",
    "CollaborativeDecisionBindingEvidenceAdoption",
]
