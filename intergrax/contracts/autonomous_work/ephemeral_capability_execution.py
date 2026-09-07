# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AW-7B ephemeral generated capability execution contracts (A1 path).

Consumes AW-7A EPHEMERAL_GENERATION_CANDIDATE decisions and correlates bounded
CodeCraft execution through a provider-neutral port. Does not mint authority,
mutate ToolRegistry, or perform durable publication.
"""

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
from intergrax.contracts.autonomous_work.profile_reference import (
    CodecraftProfileRef,
    validate_codecraft_profile_ref,
)
from intergrax.contracts.autonomous_work.references import (
    ProblemReference,
    WorkReference,
    validate_problem_reference,
    validate_work_reference,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    validate_attempt_id,
    validate_run_id,
)

EPHEMERAL_EXECUTION_POLICY_VERSION: str = "aw-7b.v1"


class WorkerEphemeralCapabilityExecutionStatus(StrEnum):
    """Typed A1 execution outcome — not durable promotion semantics."""

    SUCCEEDED = "SUCCEEDED"
    DENIED = "DENIED"
    PENDING_HITL = "PENDING_HITL"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    CONFLICT = "CONFLICT"


class WorkerEphemeralCapabilityExecutionReasonCode(StrEnum):
    """Evidence-bearing A1 execution reason codes."""

    A1_EXECUTION_SUCCEEDED = "A1_EXECUTION_SUCCEEDED"
    A1_ELIGIBILITY_REJECTED = "A1_ELIGIBILITY_REJECTED"
    DISPOSITION_MISMATCH = "DISPOSITION_MISMATCH"
    CANDIDATE_KIND_MISMATCH = "CANDIDATE_KIND_MISMATCH"
    AUTONOMY_MISMATCH = "AUTONOMY_MISMATCH"
    RISK_CLASS_MISMATCH = "RISK_CLASS_MISMATCH"
    CORRELATION_CONFLICT = "CORRELATION_CONFLICT"
    CANDIDATE_DECISION_MISMATCH = "CANDIDATE_DECISION_MISMATCH"
    PROVIDER_DENIED = "PROVIDER_DENIED"
    PROVIDER_PENDING_HITL = "PROVIDER_PENDING_HITL"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_CONFLICT = "PROVIDER_CONFLICT"


@dataclass(frozen=True, slots=True)
class WorkerEphemeralCapabilityExecutionCorrelation:
    """Runtime correlation for CodeCraft ownership — not authority."""

    tenant_id: str
    task_id: str
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    work_ref: WorkReference | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            require_non_empty_text(self.tenant_id, label="tenant_id"),
        )
        object.__setattr__(
            self,
            "task_id",
            require_non_empty_text(self.task_id, label="task_id"),
        )
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)
        if self.work_ref is not None:
            validate_work_reference(self.work_ref)


@dataclass(frozen=True, slots=True)
class WorkerEphemeralCapabilityReference:
    """Opaque ephemeral capability handle scoped to one recovery context."""

    craft_id: str
    ephemeral_tool_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "craft_id",
            require_non_empty_text(self.craft_id, label="craft_id"),
        )
        if self.ephemeral_tool_id is not None:
            object.__setattr__(
                self,
                "ephemeral_tool_id",
                require_non_empty_text(self.ephemeral_tool_id, label="ephemeral_tool_id"),
            )


@dataclass(frozen=True, slots=True)
class WorkerEphemeralCapabilityExecutionRequest:
    """Bounded A1 execution request — no authority minting fields.

    Repeated requests may mint a new ``craft_id`` unless the provider supplies
    exactly-once semantics; AW-7B does not guarantee exactly-once execution.
    """

    worker_instance_id: WorkerInstanceId
    acquisition_decision: WorkerCapabilityAcquisitionDecision
    recovery_decision_id: str
    obstacle_id: str
    need_id: str
    selected_candidate: WorkerCapabilityCandidate
    codecraft_profile_ref: CodecraftProfileRef
    generation_goal: str
    required_operations: tuple[str, ...]
    correlation: WorkerEphemeralCapabilityExecutionCorrelation
    requested_at: datetime
    constraints: str = ""
    evidence_refs: tuple[ProblemReference, ...] = ()
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        if type(self.acquisition_decision) is not WorkerCapabilityAcquisitionDecision:
            raise TypeError("acquisition_decision must be WorkerCapabilityAcquisitionDecision")
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
            "obstacle_id",
            require_non_empty_text(self.obstacle_id, label="obstacle_id"),
        )
        object.__setattr__(
            self,
            "need_id",
            require_non_empty_text(self.need_id, label="need_id"),
        )
        if type(self.selected_candidate) is not WorkerCapabilityCandidate:
            raise TypeError("selected_candidate must be WorkerCapabilityCandidate")
        validate_codecraft_profile_ref(self.codecraft_profile_ref)
        object.__setattr__(
            self,
            "generation_goal",
            require_non_empty_text(self.generation_goal, label="generation_goal"),
        )
        frozen_ops = freeze_tuple(self.required_operations, label="required_operations")
        if not frozen_ops:
            raise ValueError("required_operations must be non-empty")
        object.__setattr__(self, "required_operations", frozen_ops)
        if type(self.correlation) is not WorkerEphemeralCapabilityExecutionCorrelation:
            raise TypeError("correlation must be WorkerEphemeralCapabilityExecutionCorrelation")
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
        if self.idempotency_key is not None:
            object.__setattr__(
                self,
                "idempotency_key",
                require_non_empty_text(self.idempotency_key, label="idempotency_key"),
            )


@dataclass(frozen=True, slots=True)
class WorkerEphemeralCapabilityExecutionResult:
    """Immutable A1 execution result — ephemeral, not ToolRegistry publication.

    On ``SUCCEEDED``, ``ephemeral_capability`` remains live under CodeCraft
    session/registry ownership for the returned ``craft_correlation``. Bounded-use
    cleanup responsibility transfers to the integration/recovery consumer; lifecycle
    disposal still occurs only through canonical CodeCraft public APIs — not AW core.
    """

    status: WorkerEphemeralCapabilityExecutionStatus
    reason_code: WorkerEphemeralCapabilityExecutionReasonCode
    worker_instance_id: WorkerInstanceId
    acquisition_decision_id: str
    need_id: str
    evidence_refs: tuple[ProblemReference, ...]
    executed_at: datetime
    execution_policy_version: str = EPHEMERAL_EXECUTION_POLICY_VERSION
    ephemeral_capability: WorkerEphemeralCapabilityReference | None = None
    craft_correlation: str | None = None
    trace_correlation: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not WorkerEphemeralCapabilityExecutionStatus:
            raise TypeError("status must be WorkerEphemeralCapabilityExecutionStatus")
        if type(self.reason_code) is not WorkerEphemeralCapabilityExecutionReasonCode:
            raise TypeError("reason_code must be WorkerEphemeralCapabilityExecutionReasonCode")
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
            "evidence_refs",
            freeze_tuple(self.evidence_refs, label="evidence_refs"),
        )
        for ref in self.evidence_refs:
            validate_problem_reference(ref)
        object.__setattr__(
            self,
            "executed_at",
            require_aware_utc(self.executed_at, label="executed_at"),
        )
        object.__setattr__(
            self,
            "execution_policy_version",
            require_non_empty_text(
                self.execution_policy_version,
                label="execution_policy_version",
            ),
        )
        if self.ephemeral_capability is not None:
            if type(self.ephemeral_capability) is not WorkerEphemeralCapabilityReference:
                raise TypeError("ephemeral_capability must be WorkerEphemeralCapabilityReference")
        if self.craft_correlation is not None:
            object.__setattr__(
                self,
                "craft_correlation",
                require_non_empty_text(self.craft_correlation, label="craft_correlation"),
            )
        if self.trace_correlation is not None:
            object.__setattr__(
                self,
                "trace_correlation",
                require_non_empty_text(self.trace_correlation, label="trace_correlation"),
            )
        if self.error_code is not None:
            object.__setattr__(
                self,
                "error_code",
                require_non_empty_text(self.error_code, label="error_code"),
            )
        _validate_result_invariants(self)


def validate_a1_ephemeral_execution_eligibility(
    request: WorkerEphemeralCapabilityExecutionRequest,
) -> WorkerEphemeralCapabilityExecutionReasonCode | None:
    """Return rejection reason when request fails A1 eligibility; else None."""

    decision = request.acquisition_decision
    candidate = request.selected_candidate

    if decision.disposition is not CapabilityAcquisitionDisposition.EPHEMERAL_GENERATION_CANDIDATE:
        return WorkerEphemeralCapabilityExecutionReasonCode.DISPOSITION_MISMATCH
    if decision.autonomy_level is not WorkerAutonomyLevel.A1_EPHEMERAL_SAFE:
        return WorkerEphemeralCapabilityExecutionReasonCode.AUTONOMY_MISMATCH
    if candidate.candidate_kind is not WorkerCapabilityCandidateKind.CODECRAFT_EPHEMERAL:
        return WorkerEphemeralCapabilityExecutionReasonCode.CANDIDATE_KIND_MISMATCH
    if candidate.risk_class is not WorkerAutonomyLevel.A1_EPHEMERAL_SAFE:
        return WorkerEphemeralCapabilityExecutionReasonCode.RISK_CLASS_MISMATCH

    if decision.worker_instance_id != request.worker_instance_id:
        return WorkerEphemeralCapabilityExecutionReasonCode.CORRELATION_CONFLICT
    if decision.need_id != request.need_id:
        return WorkerEphemeralCapabilityExecutionReasonCode.CORRELATION_CONFLICT
    if decision.recovery_decision_id != request.recovery_decision_id:
        return WorkerEphemeralCapabilityExecutionReasonCode.CORRELATION_CONFLICT
    if decision.obstacle_id != request.obstacle_id:
        return WorkerEphemeralCapabilityExecutionReasonCode.CORRELATION_CONFLICT

    selected = decision.selected_candidate
    if selected is None:
        return WorkerEphemeralCapabilityExecutionReasonCode.CANDIDATE_DECISION_MISMATCH
    if selected.candidate_id != candidate.candidate_id:
        return WorkerEphemeralCapabilityExecutionReasonCode.CANDIDATE_DECISION_MISMATCH
    if selected.candidate_kind != candidate.candidate_kind:
        return WorkerEphemeralCapabilityExecutionReasonCode.CANDIDATE_DECISION_MISMATCH

    return None


def _validate_result_invariants(result: WorkerEphemeralCapabilityExecutionResult) -> None:
    if result.status is WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED:
        if result.ephemeral_capability is None:
            raise ValueError("SUCCEEDED requires ephemeral_capability")
        if result.craft_correlation is None:
            raise ValueError("SUCCEEDED requires craft_correlation")
