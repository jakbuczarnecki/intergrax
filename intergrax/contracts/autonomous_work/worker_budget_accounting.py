# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker accounting windows and budget admission contracts (AW-5B).

Worker-level daily/monthly/concurrency accounting constrains aggregate execution
activity over time. Per-Run execution token/tool accounting remains owned by the
canonical Unified Execution Runtime ``ExecutionBudgetLedger``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.autonomous_work._validation import (
    require_aware_utc,
    require_non_empty_text,
    require_non_negative_int,
)
from intergrax.contracts.autonomous_work.execution_dispatch import (
    WorkerExecutionSourceKind,
)
from intergrax.contracts.autonomous_work.ids import (
    WorkerInstanceId,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    BudgetProfileRef,
    validate_budget_profile_ref,
)
from intergrax.contracts.execution_identity import (
    ExecutionId,
    validate_execution_id,
)
from intergrax.runtime.execution.budget.models import BudgetUsageTotals


class WorkerAccountingWindowKind(StrEnum):
    """Supported immutable accounting window identities (UTC boundaries)."""

    DAILY = "daily"
    MONTHLY = "monthly"


class WorkerExecutionReservationState(StrEnum):
    """Minimum durable reservation lifecycle for concurrency accounting."""

    RESERVED = "reserved"
    BOUND_TO_EXECUTION = "bound_to_execution"
    RELEASED = "released"


class WorkerBudgetAdmissionDisposition(StrEnum):
    """Typed worker budget admission outcome."""

    ALLOWED = "allowed"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"
    CONFLICT = "conflict"


class WorkerBudgetAdmissionReason(StrEnum):
    """Implemented fail-closed budget admission reasons only."""

    DAILY_LIMIT_EXCEEDED = "daily_limit_exceeded"
    MONTHLY_LIMIT_EXCEEDED = "monthly_limit_exceeded"
    CONCURRENCY_LIMIT_EXCEEDED = "concurrency_limit_exceeded"
    RECOVERY_LIMIT_EXCEEDED = "recovery_limit_exceeded"
    CODECRAFT_LIMIT_EXCEEDED = "codecraft_limit_exceeded"
    PROACTIVE_LIMIT_EXCEEDED = "proactive_limit_exceeded"
    PROFILE_UNAVAILABLE = "profile_unavailable"
    ACCOUNTING_UNAVAILABLE = "accounting_unavailable"


@dataclass(frozen=True, slots=True)
class WorkerBudgetPolicy:
    """Immutable worker-level budget policy resolved from ``BudgetProfileRef``.

    ``None`` on a limit dimension means explicitly unlimited for that dimension.
    Missing policy resolution is not unlimited — see ``WorkerBudgetProfileResolver``.
    """

    daily_execution_limit: int | None = None
    monthly_execution_limit: int | None = None
    max_concurrent_executions: int | None = None
    daily_recovery_execution_limit: int | None = None
    daily_codecraft_execution_limit: int | None = None
    daily_proactive_evaluation_limit: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "daily_execution_limit",
            "monthly_execution_limit",
            "max_concurrent_executions",
            "daily_recovery_execution_limit",
            "daily_codecraft_execution_limit",
            "daily_proactive_evaluation_limit",
        ):
            value = getattr(self, field_name)
            if value is not None:
                require_non_negative_int(value, label=field_name)


@dataclass(frozen=True, slots=True)
class WorkerAccountingWindow:
    """Typed immutable UTC accounting window identity."""

    worker_instance_id: WorkerInstanceId
    window_kind: WorkerAccountingWindowKind
    window_start: datetime
    window_end: datetime

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        if type(self.window_kind) is not WorkerAccountingWindowKind:
            raise TypeError("window_kind must be WorkerAccountingWindowKind")
        window_start = require_aware_utc(self.window_start, label="window_start")
        window_end = require_aware_utc(self.window_end, label="window_end")
        if window_end <= window_start:
            raise ValueError("window_end must be after window_start")
        object.__setattr__(self, "window_start", window_start)
        object.__setattr__(self, "window_end", window_end)


@dataclass(frozen=True, slots=True)
class WorkerLogicalDispatchRef:
    """AW-5A logical dispatch identity for idempotent reservation."""

    worker_instance_id: WorkerInstanceId
    source_kind: WorkerExecutionSourceKind
    source_ref: str

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        if type(self.source_kind) is not WorkerExecutionSourceKind:
            raise TypeError("source_kind must be WorkerExecutionSourceKind")
        object.__setattr__(
            self,
            "source_ref",
            require_non_empty_text(self.source_ref, label="source_ref"),
        )


@dataclass(frozen=True, slots=True)
class WorkerAccountingState:
    """Durable typed accounting state for one immutable window identity."""

    window: WorkerAccountingWindow
    revision: int
    execution_count: int
    reserved_dispatch_count: int
    recovery_execution_count: int
    codecraft_execution_count: int
    proactive_evaluation_count: int
    aggregate_usage: BudgetUsageTotals

    def __post_init__(self) -> None:
        if type(self.window) is not WorkerAccountingWindow:
            raise TypeError("window must be WorkerAccountingWindow")
        require_non_negative_int(self.revision, label="revision")
        require_non_negative_int(self.execution_count, label="execution_count")
        require_non_negative_int(
            self.reserved_dispatch_count,
            label="reserved_dispatch_count",
        )
        require_non_negative_int(
            self.recovery_execution_count,
            label="recovery_execution_count",
        )
        require_non_negative_int(
            self.codecraft_execution_count,
            label="codecraft_execution_count",
        )
        require_non_negative_int(
            self.proactive_evaluation_count,
            label="proactive_evaluation_count",
        )
        if type(self.aggregate_usage) is not BudgetUsageTotals:
            raise TypeError("aggregate_usage must be BudgetUsageTotals")


@dataclass(frozen=True, slots=True)
class WorkerExecutionReservation:
    """Accounting-only execution slot reservation — not a second Execution model."""

    logical_dispatch: WorkerLogicalDispatchRef
    budget_profile_ref: BudgetProfileRef
    daily_window: WorkerAccountingWindow
    monthly_window: WorkerAccountingWindow
    reserved_at: datetime
    state: WorkerExecutionReservationState
    execution_id: ExecutionId | None = None
    bound_at: datetime | None = None
    released_at: datetime | None = None

    def __post_init__(self) -> None:
        if type(self.logical_dispatch) is not WorkerLogicalDispatchRef:
            raise TypeError("logical_dispatch must be WorkerLogicalDispatchRef")
        validate_budget_profile_ref(self.budget_profile_ref)
        if type(self.daily_window) is not WorkerAccountingWindow:
            raise TypeError("daily_window must be WorkerAccountingWindow")
        if type(self.monthly_window) is not WorkerAccountingWindow:
            raise TypeError("monthly_window must be WorkerAccountingWindow")
        object.__setattr__(
            self,
            "reserved_at",
            require_aware_utc(self.reserved_at, label="reserved_at"),
        )
        if type(self.state) is not WorkerExecutionReservationState:
            raise TypeError("state must be WorkerExecutionReservationState")
        if self.execution_id is not None:
            validate_execution_id(self.execution_id)
        if self.bound_at is not None:
            object.__setattr__(
                self,
                "bound_at",
                require_aware_utc(self.bound_at, label="bound_at"),
            )
        if self.released_at is not None:
            object.__setattr__(
                self,
                "released_at",
                require_aware_utc(self.released_at, label="released_at"),
            )


@dataclass(frozen=True, slots=True)
class WorkerBudgetAdmissionEvidence:
    """Typed admission decision evidence — no arbitrary metadata bag."""

    worker_instance_id: WorkerInstanceId
    budget_profile_ref: BudgetProfileRef
    daily_window: WorkerAccountingWindow
    monthly_window: WorkerAccountingWindow
    applied_policy: WorkerBudgetPolicy
    daily_state: WorkerAccountingState
    monthly_state: WorkerAccountingState
    active_reservation_count: int
    evaluated_at: datetime
    reason: WorkerBudgetAdmissionReason | None = None

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        validate_budget_profile_ref(self.budget_profile_ref)
        if type(self.daily_window) is not WorkerAccountingWindow:
            raise TypeError("daily_window must be WorkerAccountingWindow")
        if type(self.monthly_window) is not WorkerAccountingWindow:
            raise TypeError("monthly_window must be WorkerAccountingWindow")
        if type(self.applied_policy) is not WorkerBudgetPolicy:
            raise TypeError("applied_policy must be WorkerBudgetPolicy")
        if type(self.daily_state) is not WorkerAccountingState:
            raise TypeError("daily_state must be WorkerAccountingState")
        if type(self.monthly_state) is not WorkerAccountingState:
            raise TypeError("monthly_state must be WorkerAccountingState")
        require_non_negative_int(
            self.active_reservation_count,
            label="active_reservation_count",
        )
        object.__setattr__(
            self,
            "evaluated_at",
            require_aware_utc(self.evaluated_at, label="evaluated_at"),
        )
        if self.reason is not None:
            if type(self.reason) is not WorkerBudgetAdmissionReason:
                raise TypeError("reason must be WorkerBudgetAdmissionReason")


@dataclass(frozen=True, slots=True)
class WorkerBudgetAdmissionResult:
    """Admission/reservation outcome for one logical dispatch attempt."""

    disposition: WorkerBudgetAdmissionDisposition
    evidence: WorkerBudgetAdmissionEvidence
    reservation: WorkerExecutionReservation | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not WorkerBudgetAdmissionDisposition:
            raise TypeError("disposition must be WorkerBudgetAdmissionDisposition")
        if type(self.evidence) is not WorkerBudgetAdmissionEvidence:
            raise TypeError("evidence must be WorkerBudgetAdmissionEvidence")
        if self.reservation is not None:
            if type(self.reservation) is not WorkerExecutionReservation:
                raise TypeError("reservation must be WorkerExecutionReservation")
        if self.disposition is WorkerBudgetAdmissionDisposition.ALLOWED:
            if self.reservation is None:
                raise ValueError("ALLOWED requires reservation")
            if self.evidence.reason is not None:
                raise ValueError("ALLOWED must not expose denial reason")
        elif self.disposition is WorkerBudgetAdmissionDisposition.DENIED:
            if self.evidence.reason is None:
                raise ValueError("DENIED requires reason")
            if self.reservation is not None:
                raise ValueError("DENIED must not expose reservation")
        elif self.disposition is WorkerBudgetAdmissionDisposition.UNAVAILABLE:
            if self.evidence.reason is not WorkerBudgetAdmissionReason.ACCOUNTING_UNAVAILABLE:
                raise ValueError("UNAVAILABLE requires ACCOUNTING_UNAVAILABLE reason")
            if self.reservation is not None:
                raise ValueError("UNAVAILABLE must not expose reservation")
        elif self.disposition is WorkerBudgetAdmissionDisposition.CONFLICT:
            if self.reservation is not None:
                raise ValueError("CONFLICT must not expose reservation")


@dataclass(frozen=True, slots=True)
class WorkerBudgetReserveRequest:
    """Repository-level atomic reserve request."""

    logical_dispatch: WorkerLogicalDispatchRef
    budget_profile_ref: BudgetProfileRef
    policy: WorkerBudgetPolicy
    source_kind: WorkerExecutionSourceKind
    reserved_at: datetime

    def __post_init__(self) -> None:
        if type(self.logical_dispatch) is not WorkerLogicalDispatchRef:
            raise TypeError("logical_dispatch must be WorkerLogicalDispatchRef")
        validate_budget_profile_ref(self.budget_profile_ref)
        if type(self.policy) is not WorkerBudgetPolicy:
            raise TypeError("policy must be WorkerBudgetPolicy")
        if type(self.source_kind) is not WorkerExecutionSourceKind:
            raise TypeError("source_kind must be WorkerExecutionSourceKind")
        object.__setattr__(
            self,
            "reserved_at",
            require_aware_utc(self.reserved_at, label="reserved_at"),
        )


@dataclass(frozen=True, slots=True)
class WorkerProactiveEvaluationAccountingRequest:
    """Separate proactive evaluation accounting — no ExecutionId exists yet."""

    worker_instance_id: WorkerInstanceId
    budget_profile_ref: BudgetProfileRef
    policy: WorkerBudgetPolicy
    evaluation_ref: str
    evaluated_at: datetime

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        validate_budget_profile_ref(self.budget_profile_ref)
        if type(self.policy) is not WorkerBudgetPolicy:
            raise TypeError("policy must be WorkerBudgetPolicy")
        object.__setattr__(
            self,
            "evaluation_ref",
            require_non_empty_text(self.evaluation_ref, label="evaluation_ref"),
        )
        object.__setattr__(
            self,
            "evaluated_at",
            require_aware_utc(self.evaluated_at, label="evaluated_at"),
        )


__all__ = [
    "BudgetUsageTotals",
    "WorkerAccountingState",
    "WorkerAccountingWindow",
    "WorkerAccountingWindowKind",
    "WorkerBudgetAdmissionDisposition",
    "WorkerBudgetAdmissionEvidence",
    "WorkerBudgetAdmissionReason",
    "WorkerBudgetAdmissionResult",
    "WorkerBudgetPolicy",
    "WorkerBudgetReserveRequest",
    "WorkerExecutionReservation",
    "WorkerExecutionReservationState",
    "WorkerLogicalDispatchRef",
    "WorkerProactiveEvaluationAccountingRequest",
]
