# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution Engine dispatch for host-available / DIRECT_REUSE bound capabilities (UCA-6C-R6-R5.8-H1-R1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
from intergrax.contracts.autonomous_work.ids import (
    WorkerInstanceId,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    derive_qualified_capability_execution_request_id,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityExecutionTarget,
)
from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
    QualifiedCapabilityExecutionDispatchResult,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_run_id,
    validate_task_id,
)


@dataclass(frozen=True, slots=True)
class ExecutionBoundCapabilityExecutionDispatchRequest:
    """Canonical EE intake after host-available binding — truthful DIRECT_REUSE provenance."""

    execution_request_id: str
    execution_target: QualifiedCapabilityExecutionTarget
    tenant_id: str
    task_id: TaskId
    worker_instance_id: WorkerInstanceId
    worker_need_id: str
    direct_reuse_operation_id: str
    binding_operation_id: str
    discovery_correlation_id: str
    host_subject_reference: str
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
            ("direct_reuse_operation_id", self.direct_reuse_operation_id),
            ("binding_operation_id", self.binding_operation_id),
            ("discovery_correlation_id", self.discovery_correlation_id),
            ("host_subject_reference", self.host_subject_reference),
        ):
            object.__setattr__(
                self,
                label,
                require_non_empty_text(value, label=label),
            )
        expected_execution = derive_qualified_capability_execution_request_id(
            resume_operation_id=self.direct_reuse_operation_id,
            binding_operation_id=self.binding_operation_id,
        )
        if self.execution_request_id != expected_execution:
            raise ValueError("execution_request_id must match derived identity")
        if (
            self.execution_target.qualified_subject_reference
            != self.host_subject_reference
        ):
            raise ValueError(
                "execution_target subject must match host_subject_reference"
            )
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


@runtime_checkable
class ExecutionBoundCapabilityExecutionDispatchPort(Protocol):
    def dispatch(
        self,
        request: ExecutionBoundCapabilityExecutionDispatchRequest,
    ) -> QualifiedCapabilityExecutionDispatchResult: ...


__all__ = [
    "ExecutionBoundCapabilityExecutionDispatchPort",
    "ExecutionBoundCapabilityExecutionDispatchRequest",
    "QualifiedCapabilityExecutionDispatchDisposition",
    "QualifiedCapabilityExecutionDispatchResult",
]
