# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host-available capability binding contracts — DIRECT_REUSE without acquisition artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work._validation import (
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
    QualifiedCapabilityBindingReasonCode,
    QualifiedCapabilityExecutionTarget,
)
from intergrax.contracts.execution_identity import TaskId, validate_task_id


def derive_worker_direct_reuse_operation_id(
    *,
    recovery_decision_id: str,
    discovery_correlation_id: str,
) -> str:
    decision_id = require_non_empty_text(
        recovery_decision_id,
        label="recovery_decision_id",
    )
    correlation_id = require_non_empty_text(
        discovery_correlation_id,
        label="discovery_correlation_id",
    )
    return f"worker-direct-reuse:{decision_id}:{correlation_id}"


def host_available_subject_reference_for_identity(
    capability_identity: CapabilityIdentityKey,
) -> str:
    key = capability_identity.sort_key
    return f"host-available:{key[0]}:{key[1]}:{key[2]}:{key[3]}"


def derive_host_available_capability_binding_operation_id(
    *,
    direct_reuse_operation_id: str,
    capability_identity: CapabilityIdentityKey,
) -> str:
    reuse_id = require_non_empty_text(
        direct_reuse_operation_id,
        label="direct_reuse_operation_id",
    )
    subject_ref = host_available_subject_reference_for_identity(capability_identity)
    return f"host-available-capability-binding:{reuse_id}:{subject_ref}"


@dataclass(frozen=True, slots=True)
class HostAvailableCapabilityBindingRequest:
    """Binding dispatch for catalog host-available capability identity."""

    binding_operation_id: str
    direct_reuse_operation_id: str
    capability_identity: CapabilityIdentityKey
    worker_need_id: str
    worker_instance_id: str
    tenant_id: str
    task_id: TaskId
    discovery_correlation_id: str
    correlation_id: str | None = None
    causation_id: str | None = None
    requested_at: datetime | None = None

    def __post_init__(self) -> None:
        expected = derive_host_available_capability_binding_operation_id(
            direct_reuse_operation_id=self.direct_reuse_operation_id,
            capability_identity=self.capability_identity,
        )
        if self.binding_operation_id != expected:
            raise ValueError("binding_operation_id must match derived identity")
        object.__setattr__(
            self,
            "direct_reuse_operation_id",
            require_non_empty_text(
                self.direct_reuse_operation_id,
                label="direct_reuse_operation_id",
            ),
        )
        if type(self.capability_identity) is not CapabilityIdentityKey:
            raise TypeError("capability_identity must be CapabilityIdentityKey")
        object.__setattr__(
            self,
            "worker_need_id",
            require_non_empty_text(self.worker_need_id, label="worker_need_id"),
        )
        object.__setattr__(
            self,
            "worker_instance_id",
            require_non_empty_text(
                self.worker_instance_id,
                label="worker_instance_id",
            ),
        )
        object.__setattr__(
            self,
            "tenant_id",
            require_non_empty_text(self.tenant_id, label="tenant_id"),
        )
        validate_task_id(self.task_id)
        object.__setattr__(
            self,
            "discovery_correlation_id",
            require_non_empty_text(
                self.discovery_correlation_id,
                label="discovery_correlation_id",
            ),
        )
        if self.requested_at is not None:
            object.__setattr__(
                self,
                "requested_at",
                require_aware_utc(self.requested_at, label="requested_at"),
            )


@dataclass(frozen=True, slots=True)
class HostAvailableCapabilityBindingResult:
    """Typed host-available binding outcome — reuses EE execution target shape."""

    binding_operation_id: str
    outcome: QualifiedCapabilityBindingOutcome
    reason_code: QualifiedCapabilityBindingReasonCode
    provider_id: str | None = None
    execution_target: QualifiedCapabilityExecutionTarget | None = None
    host_subject_reference: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    reason_detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binding_operation_id",
            require_non_empty_text(
                self.binding_operation_id,
                label="binding_operation_id",
            ),
        )
        if type(self.outcome) is not QualifiedCapabilityBindingOutcome:
            raise TypeError("outcome must be QualifiedCapabilityBindingOutcome")
        if type(self.reason_code) is not QualifiedCapabilityBindingReasonCode:
            raise TypeError("reason_code must be QualifiedCapabilityBindingReasonCode")
        if self.outcome is QualifiedCapabilityBindingOutcome.BOUND:
            if self.execution_target is None:
                raise ValueError("BOUND requires execution_target")
            if self.host_subject_reference is None:
                raise ValueError("BOUND requires host_subject_reference")


@runtime_checkable
class HostAvailableCapabilityBindingPort(Protocol):
    """Host-available binding coordination — plugin providers behind one owner."""

    def bind(
        self,
        request: HostAvailableCapabilityBindingRequest,
    ) -> HostAvailableCapabilityBindingResult: ...


@runtime_checkable
class HostAvailableCapabilityBindingProvider(Protocol):
    """Plugin host-available binding — no acquisition or qualification artifacts."""

    @property
    def provider_id(self) -> str: ...

    def supports(self, request: HostAvailableCapabilityBindingRequest) -> bool: ...

    def bind(
        self,
        request: HostAvailableCapabilityBindingRequest,
    ) -> HostAvailableCapabilityBindingResult: ...


__all__ = [
    "derive_worker_direct_reuse_operation_id",
    "HostAvailableCapabilityBindingPort",
    "HostAvailableCapabilityBindingProvider",
    "HostAvailableCapabilityBindingRequest",
    "HostAvailableCapabilityBindingResult",
    "derive_host_available_capability_binding_operation_id",
    "host_available_subject_reference_for_identity",
]
