# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Post-qualification worker capability resume contracts (UCA-6C)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.contracts.autonomous_work._validation import (
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.execution_authority import (
    validate_authority_scopes,
)
from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.autonomous_work.ids import (
    WorkerInstanceId,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingResult,
    QualifiedCapabilityExecutionTarget,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)


class WorkerQualifiedCapabilityExecutionDisposition(StrEnum):
    """Execution Engine handoff outcome — distinct from qualification."""

    DISPATCHED = "dispatched"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    REJECTED = "rejected"


class WorkerQualifiedCapabilityResumeOutcome(StrEnum):
    """Resume stage outcome — QUALIFIED is not automatically executable."""

    EXECUTION_DISPATCHED = "execution_dispatched"
    BINDING_UNAVAILABLE = "binding_unavailable"
    BINDING_BLOCKED = "binding_blocked"
    BINDING_FAILED = "binding_failed"
    BINDING_HITL = "binding_hitl"
    QUALIFICATION_NOT_ELIGIBLE = "qualification_not_eligible"
    EXECUTION_UNAVAILABLE = "execution_unavailable"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_REJECTED = "execution_rejected"


def derive_worker_capability_resume_operation_id(
    *,
    recovery_decision_id: str,
    qualification_request_id: str,
) -> str:
    decision_id = require_non_empty_text(
        recovery_decision_id,
        label="recovery_decision_id",
    )
    qual_id = require_non_empty_text(
        qualification_request_id,
        label="qualification_request_id",
    )
    return f"worker-capability-resume:{decision_id}:{qual_id}"


def derive_qualified_capability_execution_request_id(
    *,
    resume_operation_id: str,
    binding_operation_id: str,
) -> str:
    resume_id = require_non_empty_text(
        resume_operation_id,
        label="resume_operation_id",
    )
    binding_id = require_non_empty_text(
        binding_operation_id,
        label="binding_operation_id",
    )
    return f"qualified-capability-execution:{resume_id}:{binding_id}"


@dataclass(frozen=True, slots=True)
class WorkerQualifiedCapabilityResumeRequest:
    """Evidence-backed resume — no discovery, acquisition, or re-qualification."""

    worker_instance_id: WorkerInstanceId
    worker_need_id: str
    recovery_decision_id: str
    provenance: WorkerCapabilityRecoveryProvenance
    acquisition_result: CapabilityAcquisitionResult
    qualification_result: CapabilityQualificationResult
    resume_operation_id: str
    tenant_id: str
    task_id: TaskId
    requested_at: datetime
    requested_authority_scopes: tuple[str, ...]
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(
            self,
            "requested_authority_scopes",
            validate_authority_scopes(self.requested_authority_scopes),
        )
        object.__setattr__(
            self,
            "worker_need_id",
            require_non_empty_text(self.worker_need_id, label="worker_need_id"),
        )
        object.__setattr__(
            self,
            "recovery_decision_id",
            require_non_empty_text(
                self.recovery_decision_id,
                label="recovery_decision_id",
            ),
        )
        if type(self.provenance) is not WorkerCapabilityRecoveryProvenance:
            raise TypeError("provenance must be WorkerCapabilityRecoveryProvenance")
        if type(self.acquisition_result) is not CapabilityAcquisitionResult:
            raise TypeError("acquisition_result must be CapabilityAcquisitionResult")
        if type(self.qualification_result) is not CapabilityQualificationResult:
            raise TypeError(
                "qualification_result must be CapabilityQualificationResult"
            )
        expected_resume = derive_worker_capability_resume_operation_id(
            recovery_decision_id=self.recovery_decision_id,
            qualification_request_id=self.qualification_result.qualification_request_id,
        )
        if self.resume_operation_id != expected_resume:
            raise ValueError("resume_operation_id must match derived identity")
        object.__setattr__(
            self,
            "tenant_id",
            require_non_empty_text(self.tenant_id, label="tenant_id"),
        )
        validate_task_id(self.task_id)
        object.__setattr__(
            self,
            "requested_at",
            require_aware_utc(self.requested_at, label="requested_at"),
        )
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)
        acquisition = self.acquisition_result
        qualification = self.qualification_result
        if qualification.acquisition_request_id != acquisition.request_id:
            raise ValueError(
                "qualification_result.acquisition_request_id must match acquisition_result.request_id",
            )
        if qualification.gap_id != acquisition.gap_id:
            raise ValueError(
                "qualification_result.gap_id must match acquisition_result.gap_id"
            )
        if qualification.strategy_id != acquisition.strategy_id:
            raise ValueError(
                "qualification_result.strategy_id must match acquisition_result.strategy_id",
            )
        provenance = self.provenance
        if provenance.acquisition_request_id != acquisition.request_id:
            raise ValueError("provenance.acquisition_request_id must match acquisition")
        if provenance.gap_id != acquisition.gap_id:
            raise ValueError("provenance.gap_id must match acquisition")
        if provenance.acquisition_strategy_id != acquisition.strategy_id:
            raise ValueError(
                "provenance.acquisition_strategy_id must match acquisition"
            )
        if (
            provenance.qualification_request_id
            != qualification.qualification_request_id
        ):
            raise ValueError(
                "provenance.qualification_request_id must match qualification"
            )


@dataclass(frozen=True, slots=True)
class WorkerQualifiedCapabilityExecutionRequest:
    """Execution Engine handoff after successful binding."""

    resume_operation_id: str
    binding_operation_id: str
    execution_request_id: str
    execution_target: QualifiedCapabilityExecutionTarget
    worker_instance_id: WorkerInstanceId
    worker_need_id: str
    tenant_id: str
    task_id: TaskId
    qualification_request_id: str
    acquisition_request_id: str
    qualified_subject_reference: str
    requested_at: datetime
    admitted_governance_identity: AdmittedRootGovernanceIdentity
    effective_authority_decision: EffectiveAuthorityDecision
    collaborative_authority_scopes: tuple[str, ...]
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resume_operation_id",
            require_non_empty_text(
                self.resume_operation_id, label="resume_operation_id"
            ),
        )
        object.__setattr__(
            self,
            "binding_operation_id",
            require_non_empty_text(
                self.binding_operation_id,
                label="binding_operation_id",
            ),
        )
        object.__setattr__(
            self,
            "execution_request_id",
            require_non_empty_text(
                self.execution_request_id,
                label="execution_request_id",
            ),
        )
        expected_execution = derive_qualified_capability_execution_request_id(
            resume_operation_id=self.resume_operation_id,
            binding_operation_id=self.binding_operation_id,
        )
        if self.execution_request_id != expected_execution:
            raise ValueError("execution_request_id must match derived identity")
        if type(self.execution_target) is not QualifiedCapabilityExecutionTarget:
            raise TypeError(
                "execution_target must be QualifiedCapabilityExecutionTarget"
            )
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(
            self,
            "worker_need_id",
            require_non_empty_text(self.worker_need_id, label="worker_need_id"),
        )
        object.__setattr__(
            self,
            "tenant_id",
            require_non_empty_text(self.tenant_id, label="tenant_id"),
        )
        validate_task_id(self.task_id)
        object.__setattr__(
            self,
            "qualification_request_id",
            require_non_empty_text(
                self.qualification_request_id,
                label="qualification_request_id",
            ),
        )
        object.__setattr__(
            self,
            "acquisition_request_id",
            require_non_empty_text(
                self.acquisition_request_id,
                label="acquisition_request_id",
            ),
        )
        object.__setattr__(
            self,
            "qualified_subject_reference",
            require_non_empty_text(
                self.qualified_subject_reference,
                label="qualified_subject_reference",
            ),
        )
        if (
            self.execution_target.qualified_subject_reference
            != self.qualified_subject_reference
        ):
            raise ValueError("execution_target subject must match request subject")
        object.__setattr__(
            self,
            "requested_at",
            require_aware_utc(self.requested_at, label="requested_at"),
        )
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)
        if (
            type(self.admitted_governance_identity)
            is not AdmittedRootGovernanceIdentity
        ):
            raise TypeError(
                "admitted_governance_identity must be AdmittedRootGovernanceIdentity",
            )
        if type(self.effective_authority_decision) is not EffectiveAuthorityDecision:
            raise TypeError(
                "effective_authority_decision must be EffectiveAuthorityDecision",
            )
        object.__setattr__(
            self,
            "collaborative_authority_scopes",
            validate_authority_scopes(self.collaborative_authority_scopes),
        )
        if self.tenant_id != self.admitted_governance_identity.tenant_id:
            raise ValueError(
                "tenant_id must match admitted_governance_identity.tenant_id",
            )


@dataclass(frozen=True, slots=True)
class WorkerQualifiedCapabilityExecutionResult:
    """Typed execution outcome — not a capability gap signal."""

    disposition: WorkerQualifiedCapabilityExecutionDisposition
    execution_request_id: str | None = None
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None
    reason_detail: str = ""

    def __post_init__(self) -> None:
        if type(self.disposition) is not WorkerQualifiedCapabilityExecutionDisposition:
            raise TypeError(
                "disposition must be WorkerQualifiedCapabilityExecutionDisposition",
            )
        if self.execution_request_id is not None:
            object.__setattr__(
                self,
                "execution_request_id",
                require_non_empty_text(
                    self.execution_request_id,
                    label="execution_request_id",
                ),
            )
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)
        if self.execution_id is not None:
            validate_execution_id(self.execution_id)
        if (
            self.disposition is WorkerQualifiedCapabilityExecutionDisposition.DISPATCHED
            and self.execution_request_id is None
        ):
            raise ValueError("DISPATCHED requires execution_request_id")


@dataclass(frozen=True, slots=True)
class WorkerQualifiedCapabilityResumeResult:
    """Post-qualification resume result with binding and optional execution evidence."""

    outcome: WorkerQualifiedCapabilityResumeOutcome
    resume_operation_id: str
    provenance: WorkerCapabilityRecoveryProvenance
    binding_result: QualifiedCapabilityBindingResult | None = None
    execution_result: WorkerQualifiedCapabilityExecutionResult | None = None
    decided_at: datetime | None = None

    def __post_init__(self) -> None:
        if type(self.outcome) is not WorkerQualifiedCapabilityResumeOutcome:
            raise TypeError("outcome must be WorkerQualifiedCapabilityResumeOutcome")
        object.__setattr__(
            self,
            "resume_operation_id",
            require_non_empty_text(
                self.resume_operation_id, label="resume_operation_id"
            ),
        )
        if type(self.provenance) is not WorkerCapabilityRecoveryProvenance:
            raise TypeError("provenance must be WorkerCapabilityRecoveryProvenance")
        if self.binding_result is not None and not isinstance(
            self.binding_result,
            QualifiedCapabilityBindingResult,
        ):
            raise TypeError("binding_result must be QualifiedCapabilityBindingResult")
        if (
            self.execution_result is not None
            and type(
                self.execution_result,
            )
            is not WorkerQualifiedCapabilityExecutionResult
        ):
            raise TypeError(
                "execution_result must be WorkerQualifiedCapabilityExecutionResult",
            )
        if self.decided_at is not None:
            object.__setattr__(
                self,
                "decided_at",
                require_aware_utc(self.decided_at, label="decided_at"),
            )


__all__ = [
    "WorkerQualifiedCapabilityExecutionDisposition",
    "WorkerQualifiedCapabilityExecutionRequest",
    "WorkerQualifiedCapabilityExecutionResult",
    "WorkerQualifiedCapabilityResumeOutcome",
    "WorkerQualifiedCapabilityResumeRequest",
    "WorkerQualifiedCapabilityResumeResult",
    "derive_qualified_capability_execution_request_id",
    "derive_worker_capability_resume_operation_id",
]
