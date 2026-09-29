# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AW-7C A2 scoped adaptive integration orchestration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.autonomous_work._validation import (
    freeze_tuple,
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
)
from intergrax.contracts.autonomous_work.ids import (
    WorkerInstanceId,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.references import (
    ProblemReference,
    validate_problem_reference,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)
from intergrax.contracts.capability_qualification.qualification_subject import (
    CapabilityQualificationSubject,
)
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationArtifact,
    ScopedIntegrationAdaptationPort,
    ScopedIntegrationAdaptationScope,
)

SCOPED_ADAPTIVE_INTEGRATION_POLICY_VERSION: str = "aw-7c.v1"


class ScopedAdaptiveIntegrationPreparationStatus(StrEnum):
    ARTIFACT_PRODUCED = "ARTIFACT_PRODUCED"
    QUALIFICATION_PENDING = "QUALIFICATION_PENDING"
    DENIED = "DENIED"
    FAILED = "FAILED"


class ScopedAdaptiveIntegrationFailureReason(StrEnum):
    A2_ELIGIBILITY_REJECTED = "A2_ELIGIBILITY_REJECTED"
    DISPOSITION_MISMATCH = "DISPOSITION_MISMATCH"
    AUTONOMY_MISMATCH = "AUTONOMY_MISMATCH"
    CANDIDATE_KIND_MISMATCH = "CANDIDATE_KIND_MISMATCH"
    RISK_CLASS_MISMATCH = "RISK_CLASS_MISMATCH"
    CORRELATION_CONFLICT = "CORRELATION_CONFLICT"
    CANDIDATE_DECISION_MISMATCH = "CANDIDATE_DECISION_MISMATCH"
    TENANT_MISMATCH = "TENANT_MISMATCH"
    OPERATIONS_INCOMPATIBLE = "OPERATIONS_INCOMPATIBLE"
    INTEGRATION_IDENTITY_MISMATCH = "INTEGRATION_IDENTITY_MISMATCH"
    ADAPTATION_FAILED = "ADAPTATION_FAILED"


@dataclass(frozen=True, slots=True)
class ScopedAdaptiveIntegrationCorrelation:
    tenant_id: str
    recovery_episode_id: str | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            require_non_empty_text(self.tenant_id, label="tenant_id"),
        )
        if self.recovery_episode_id is not None:
            object.__setattr__(
                self,
                "recovery_episode_id",
                require_non_empty_text(
                    self.recovery_episode_id,
                    label="recovery_episode_id",
                ),
            )
        if self.correlation_id is not None:
            object.__setattr__(
                self,
                "correlation_id",
                require_non_empty_text(self.correlation_id, label="correlation_id"),
            )


@dataclass(frozen=True, slots=True)
class ScopedAdaptiveIntegrationExecutionRequest:
    worker_instance_id: WorkerInstanceId
    correlation: ScopedAdaptiveIntegrationCorrelation
    acquisition_decision: WorkerCapabilityAcquisitionDecision
    selected_candidate: WorkerCapabilityCandidate
    need_id: str
    recovery_decision_id: str
    integration_capability_ref: str
    required_operations: tuple[str, ...]
    adaptation_scope: ScopedIntegrationAdaptationScope
    requested_at: datetime
    evidence_refs: tuple[ProblemReference, ...] = ()
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        if type(self.correlation) is not ScopedAdaptiveIntegrationCorrelation:
            raise TypeError("correlation must be ScopedAdaptiveIntegrationCorrelation")
        if type(self.acquisition_decision) is not WorkerCapabilityAcquisitionDecision:
            raise TypeError("acquisition_decision must be WorkerCapabilityAcquisitionDecision")
        if type(self.selected_candidate) is not WorkerCapabilityCandidate:
            raise TypeError("selected_candidate must be WorkerCapabilityCandidate")
        object.__setattr__(
            self,
            "need_id",
            require_non_empty_text(self.need_id, label="need_id"),
        )
        object.__setattr__(
            self,
            "recovery_decision_id",
            require_non_empty_text(
                self.recovery_decision_id,
                label="recovery_decision_id",
            ),
        )
        object.__setattr__(
            self,
            "integration_capability_ref",
            require_non_empty_text(
                self.integration_capability_ref,
                label="integration_capability_ref",
            ),
        )
        frozen_ops = freeze_tuple(self.required_operations, label="required_operations")
        if not frozen_ops:
            raise ValueError("required_operations must be non-empty")
        object.__setattr__(self, "required_operations", frozen_ops)
        if type(self.adaptation_scope) is not ScopedIntegrationAdaptationScope:
            raise TypeError("adaptation_scope must be ScopedIntegrationAdaptationScope")
        object.__setattr__(
            self,
            "requested_at",
            require_aware_utc(self.requested_at, label="requested_at"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            freeze_tuple(self.evidence_refs, label="evidence_refs"),
        )
        for ref in self.evidence_refs:
            validate_problem_reference(ref)
        if self.adaptation_scope.tenant_id != self.correlation.tenant_id:
            raise ValueError("adaptation scope tenant must match correlation tenant")


@dataclass(frozen=True, slots=True)
class ScopedAdaptiveIntegrationPreparationResult:
    status: ScopedAdaptiveIntegrationPreparationStatus
    reason_code: ScopedAdaptiveIntegrationFailureReason | None
    worker_instance_id: WorkerInstanceId
    acquisition_decision_id: str
    need_id: str
    prepared_at: datetime
    policy_version: str = SCOPED_ADAPTIVE_INTEGRATION_POLICY_VERSION
    artifact: ScopedIntegrationAdaptationArtifact | None = None
    qualification_subject: CapabilityQualificationSubject | None = None
    qualification_request: CapabilityQualificationRequest | None = None
    evidence_refs: tuple[ProblemReference, ...] = ()
    error_detail: str = ""

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(
            self,
            "acquisition_decision_id",
            require_non_empty_text(
                self.acquisition_decision_id,
                label="acquisition_decision_id",
            ),
        )
        object.__setattr__(
            self,
            "need_id",
            require_non_empty_text(self.need_id, label="need_id"),
        )
        object.__setattr__(
            self,
            "prepared_at",
            require_aware_utc(self.prepared_at, label="prepared_at"),
        )
        if self.status is ScopedAdaptiveIntegrationPreparationStatus.QUALIFICATION_PENDING:
            if self.artifact is None:
                raise ValueError("QUALIFICATION_PENDING requires artifact")
            if self.qualification_subject is None:
                raise ValueError("QUALIFICATION_PENDING requires qualification_subject")
            if self.qualification_request is None:
                raise ValueError("QUALIFICATION_PENDING requires qualification_request")


WorkerScopedAdaptiveIntegrationOrchestrationPort = ScopedIntegrationAdaptationPort


def validate_a2_scoped_adaptive_integration_eligibility(
    request: ScopedAdaptiveIntegrationExecutionRequest,
) -> ScopedAdaptiveIntegrationFailureReason | None:
    decision = request.acquisition_decision
    candidate = request.selected_candidate
    if decision.disposition is not CapabilityAcquisitionDisposition.SCOPED_ADAPTATION_CANDIDATE:
        return ScopedAdaptiveIntegrationFailureReason.DISPOSITION_MISMATCH
    if decision.autonomy_level is not WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE:
        return ScopedAdaptiveIntegrationFailureReason.AUTONOMY_MISMATCH
    if candidate.candidate_kind is not WorkerCapabilityCandidateKind.ADAPTIVE_INTEGRATION:
        return ScopedAdaptiveIntegrationFailureReason.CANDIDATE_KIND_MISMATCH
    if candidate.risk_class is not WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE:
        return ScopedAdaptiveIntegrationFailureReason.RISK_CLASS_MISMATCH
    if decision.worker_instance_id != request.worker_instance_id:
        return ScopedAdaptiveIntegrationFailureReason.CORRELATION_CONFLICT
    if decision.need_id != request.need_id:
        return ScopedAdaptiveIntegrationFailureReason.CORRELATION_CONFLICT
    if decision.recovery_decision_id != request.recovery_decision_id:
        return ScopedAdaptiveIntegrationFailureReason.CORRELATION_CONFLICT
    selected = decision.selected_candidate
    if selected is None:
        return ScopedAdaptiveIntegrationFailureReason.CANDIDATE_DECISION_MISMATCH
    if selected.candidate_id != candidate.candidate_id:
        return ScopedAdaptiveIntegrationFailureReason.CANDIDATE_DECISION_MISMATCH
    if selected.candidate_kind != candidate.candidate_kind:
        return ScopedAdaptiveIntegrationFailureReason.CANDIDATE_DECISION_MISMATCH
    if not request.correlation.tenant_id:
        return ScopedAdaptiveIntegrationFailureReason.TENANT_MISMATCH
    scope = request.adaptation_scope
    if scope.tenant_id != request.correlation.tenant_id:
        return ScopedAdaptiveIntegrationFailureReason.TENANT_MISMATCH
    if candidate.capability_ref != request.integration_capability_ref:
        return ScopedAdaptiveIntegrationFailureReason.INTEGRATION_IDENTITY_MISMATCH
    candidate_ops = frozenset(candidate.operations)
    if not all(op in candidate_ops for op in request.required_operations):
        return ScopedAdaptiveIntegrationFailureReason.OPERATIONS_INCOMPATIBLE
    scope_op_values = frozenset(op.value for op in scope.permitted_operations)
    if not all(op in scope_op_values for op in request.required_operations):
        return ScopedAdaptiveIntegrationFailureReason.OPERATIONS_INCOMPATIBLE
    if scope.candidate_id != candidate.candidate_id:
        return ScopedAdaptiveIntegrationFailureReason.INTEGRATION_IDENTITY_MISMATCH
    return None


__all__ = [
    "SCOPED_ADAPTIVE_INTEGRATION_POLICY_VERSION",
    "ScopedAdaptiveIntegrationCorrelation",
    "ScopedAdaptiveIntegrationExecutionRequest",
    "ScopedAdaptiveIntegrationFailureReason",
    "ScopedAdaptiveIntegrationPreparationResult",
    "ScopedAdaptiveIntegrationPreparationStatus",
    "WorkerScopedAdaptiveIntegrationOrchestrationPort",
    "validate_a2_scoped_adaptive_integration_eligibility",
]
