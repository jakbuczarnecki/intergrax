# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker capability fulfillment outcomes — consumer orchestration surface (UCA-6C-R6-R5.8)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.autonomous_work._validation import (
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    WorkerCapabilityAcquisitionRequest,
)
from intergrax.contracts.autonomous_work.ids import (
    WorkerInstanceId,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionResult,
    WorkerQualifiedCapabilityResumeResult,
)
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_run_id,
    validate_task_id,
)


class WorkerCapabilityFulfillmentDisposition(StrEnum):
    """Terminal consumer fulfillment disposition — not a duplicate discovery owner."""

    EXECUTION_DISPATCHED = "execution_dispatched"
    CAPABILITY_GAP = "capability_gap"
    DISCOVERY_BLOCKED = "discovery_blocked"
    DISCOVERY_UNAVAILABLE = "discovery_unavailable"
    DISCOVERY_INCOMPLETE = "discovery_incomplete"
    DISCOVERY_CONFLICT = "discovery_conflict"
    REALIZATION_FAILED = "realization_failed"
    REALIZATION_NOT_VISIBLE = "realization_not_visible"
    QUALIFICATION_FAILED = "qualification_failed"
    BINDING_FAILED = "binding_failed"
    EXECUTION_FAILED = "execution_failed"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True, slots=True)
class WorkerCapabilityFulfillmentRequest:
    """Single logical consumer fulfillment invocation — one discovery owner entry."""

    acquisition_request: WorkerCapabilityAcquisitionRequest
    worker_instance_id: WorkerInstanceId
    tenant_id: str
    task_id: TaskId
    requested_at: datetime
    requested_authority_scopes: tuple[str, ...]
    allow_generic_acquisition: bool = True
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
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


@dataclass(frozen=True, slots=True)
class WorkerCapabilityFulfillmentResult:
    """Evidence-bearing fulfillment result for worker consumer orchestration."""

    disposition: WorkerCapabilityFulfillmentDisposition
    provenance: WorkerCapabilityRecoveryProvenance
    recovery_outcome: WorkerCapabilityRecoveryOutcome | None = None
    capability_gap: CapabilityGap | None = None
    resume_result: WorkerQualifiedCapabilityResumeResult | None = None
    execution_result: WorkerQualifiedCapabilityExecutionResult | None = None
    decided_at: datetime | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not WorkerCapabilityFulfillmentDisposition:
            raise TypeError(
                "disposition must be WorkerCapabilityFulfillmentDisposition"
            )
        if type(self.provenance) is not WorkerCapabilityRecoveryProvenance:
            raise TypeError("provenance must be WorkerCapabilityRecoveryProvenance")
        if self.decided_at is not None:
            object.__setattr__(
                self,
                "decided_at",
                require_aware_utc(self.decided_at, label="decided_at"),
            )


__all__ = [
    "WorkerCapabilityFulfillmentDisposition",
    "WorkerCapabilityFulfillmentRequest",
    "WorkerCapabilityFulfillmentResult",
]
