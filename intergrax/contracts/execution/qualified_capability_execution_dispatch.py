# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public Execution Engine dispatch contract for bound qualified capabilities (UCA-6C-R)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

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
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
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


class QualifiedCapabilityExecutionDispatchDisposition(StrEnum):
    """Typed dispatch outcome — not a capability gap signal."""

    DISPATCHED = "dispatched"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class QualifiedCapabilityExecutionDispatchRequest:
    """Canonical Execution Engine intake for one bound qualified capability."""

    execution_request_id: str
    execution_target: QualifiedCapabilityExecutionTarget
    tenant_id: str
    task_id: TaskId
    worker_instance_id: WorkerInstanceId
    worker_need_id: str
    resume_operation_id: str
    binding_operation_id: str
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
            "execution_request_id",
            require_non_empty_text(
                self.execution_request_id,
                label="execution_request_id",
            ),
        )
        if type(self.execution_target) is not QualifiedCapabilityExecutionTarget:
            raise TypeError(
                "execution_target must be QualifiedCapabilityExecutionTarget"
            )
        object.__setattr__(
            self,
            "tenant_id",
            require_non_empty_text(self.tenant_id, label="tenant_id"),
        )
        validate_task_id(self.task_id)
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(
            self,
            "worker_need_id",
            require_non_empty_text(self.worker_need_id, label="worker_need_id"),
        )
        for label, value in (
            ("resume_operation_id", self.resume_operation_id),
            ("binding_operation_id", self.binding_operation_id),
            ("qualification_request_id", self.qualification_request_id),
            ("acquisition_request_id", self.acquisition_request_id),
            ("qualified_subject_reference", self.qualified_subject_reference),
        ):
            object.__setattr__(
                self,
                label,
                require_non_empty_text(value, label=label),
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
class QualifiedCapabilityExecutionDispatchResult:
    disposition: QualifiedCapabilityExecutionDispatchDisposition
    execution_request_id: str | None = None
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None
    reason_detail: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.disposition)
            is not QualifiedCapabilityExecutionDispatchDisposition
        ):
            raise TypeError(
                "disposition must be QualifiedCapabilityExecutionDispatchDisposition",
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
        if (
            self.disposition
            is QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED
        ):
            if self.execution_request_id is None:
                raise ValueError("DISPATCHED requires execution_request_id")
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)
        if self.execution_id is not None:
            validate_execution_id(self.execution_id)


@runtime_checkable
class QualifiedCapabilityExecutionDispatchPort(Protocol):
    """Ingress dedup port — delegates to canonical ExecutionRuntime via root launch."""

    def dispatch(
        self,
        request: QualifiedCapabilityExecutionDispatchRequest,
    ) -> QualifiedCapabilityExecutionDispatchResult: ...


__all__ = [
    "QualifiedCapabilityExecutionDispatchDisposition",
    "QualifiedCapabilityExecutionDispatchPort",
    "QualifiedCapabilityExecutionDispatchRequest",
    "QualifiedCapabilityExecutionDispatchResult",
]
