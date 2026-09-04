# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded proactive WorkerGoal evaluation contracts (AW-4B)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.autonomous_work._validation import (
    freeze_tuple,
    require_aware_utc,
    require_non_empty_text,
    require_non_negative_int,
)
from intergrax.contracts.autonomous_work.ids import (
    WakeUpId,
    WorkerGoalId,
    WorkerInstanceId,
    validate_wake_up_id,
    validate_worker_goal_id,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.revision import Revision, validate_revision
from intergrax.contracts.autonomous_work.references import (
    ProgressProjectionRef,
    validate_progress_projection_ref,
)
from intergrax.contracts.autonomous_work.wake_up import (
    WorkerWakeUpContext,
    WorkerWakeUpDisposition,
)


class GoalEvaluationDisposition(StrEnum):
    """Typed outcome for one bounded goal evaluation."""

    NO_ACTION = "NO_ACTION"
    ACTION_REQUIRED = "ACTION_REQUIRED"
    NOT_DUE = "NOT_DUE"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class GoalEvaluationReasonCode(StrEnum):
    """Evidence-bearing reason codes supported by deterministic evaluators."""

    SLA_RISK = "SLA_RISK"
    DEADLINE_RISK = "DEADLINE_RISK"
    SUCCESS_CRITERIA_NOT_MET = "SUCCESS_CRITERIA_NOT_MET"
    PROGRESS_STALLED = "PROGRESS_STALLED"
    THRESHOLD_BREACH = "THRESHOLD_BREACH"
    PROGRESS_PROJECTION_UNAVAILABLE = "PROGRESS_PROJECTION_UNAVAILABLE"
    PROGRESS_PROJECTION_STALE = "PROGRESS_PROJECTION_STALE"
    CADENCE_POLICY_UNAVAILABLE = "CADENCE_POLICY_UNAVAILABLE"
    OPEN_WORK_ALREADY_PENDING = "OPEN_WORK_ALREADY_PENDING"
    CRITERIA_MET = "CRITERIA_MET"


@dataclass(frozen=True, slots=True)
class GoalEvaluationBatchLimit:
    """Hard per-wake-up evaluation bound."""

    max_goals: int

    def __post_init__(self) -> None:
        value = require_non_negative_int(self.max_goals, label="max_goals")
        if value <= 0:
            raise ValueError("max_goals must be positive")


@dataclass(frozen=True, slots=True)
class GoalEvaluationCadenceState:
    """Durable last-evaluation marker for proactive cadence eligibility."""

    goal_id: WorkerGoalId
    last_evaluated_at: datetime
    revision: Revision

    def __post_init__(self) -> None:
        validate_worker_goal_id(self.goal_id)
        object.__setattr__(
            self,
            "last_evaluated_at",
            require_aware_utc(self.last_evaluated_at, label="last_evaluated_at"),
        )
        validate_revision(self.revision)


@dataclass(frozen=True, slots=True)
class GoalEvaluationCadencePolicy:
    """Resolved cadence policy — answers eligibility, not scheduling."""

    minimum_interval_seconds: int

    def __post_init__(self) -> None:
        value = require_non_negative_int(
            self.minimum_interval_seconds,
            label="minimum_interval_seconds",
        )
        if value <= 0:
            raise ValueError("minimum_interval_seconds must be positive")


@dataclass(frozen=True, slots=True)
class GoalProgressProjection:
    """Typed progress projection resolved from ``ProgressProjectionRef``."""

    projection_ref: ProgressProjectionRef
    observed_at: datetime | None = None
    current_value: float | None = None
    target_value: float | None = None
    status: str | None = None
    evidence_refs: tuple[str, ...] = ()
    stale_after_seconds: int | None = None

    def __post_init__(self) -> None:
        validate_progress_projection_ref(self.projection_ref)
        if self.observed_at is not None:
            object.__setattr__(
                self,
                "observed_at",
                require_aware_utc(self.observed_at, label="observed_at"),
            )
        object.__setattr__(
            self,
            "evidence_refs",
            freeze_tuple(self.evidence_refs, label="evidence_refs"),
        )
        if self.stale_after_seconds is not None:
            stale_value = require_non_negative_int(
                self.stale_after_seconds,
                label="stale_after_seconds",
            )
            if stale_value <= 0:
                raise ValueError("stale_after_seconds must be positive")


@dataclass(frozen=True, slots=True)
class GoalEvaluationDecision:
    """Immutable evaluation outcome for one goal."""

    goal_id: WorkerGoalId
    goal_revision: Revision
    evaluated_at: datetime
    disposition: GoalEvaluationDisposition
    reason: str
    wake_up_id: WakeUpId
    reason_code: GoalEvaluationReasonCode | None = None
    evidence_refs: tuple[str, ...] = ()
    progress_projection_ref: ProgressProjectionRef | None = None

    def __post_init__(self) -> None:
        validate_worker_goal_id(self.goal_id)
        validate_revision(self.goal_revision)
        object.__setattr__(
            self,
            "evaluated_at",
            require_aware_utc(self.evaluated_at, label="evaluated_at"),
        )
        if type(self.disposition) is not GoalEvaluationDisposition:
            raise TypeError("disposition must be GoalEvaluationDisposition")
        object.__setattr__(
            self,
            "reason",
            require_non_empty_text(self.reason, label="reason"),
        )
        validate_wake_up_id(self.wake_up_id)
        if self.reason_code is not None:
            if type(self.reason_code) is not GoalEvaluationReasonCode:
                raise TypeError("reason_code must be GoalEvaluationReasonCode")
        object.__setattr__(
            self,
            "evidence_refs",
            freeze_tuple(self.evidence_refs, label="evidence_refs"),
        )
        if self.progress_projection_ref is not None:
            validate_progress_projection_ref(self.progress_projection_ref)


@dataclass(frozen=True, slots=True)
class WorkerGoalEvaluationRequest:
    """Bounded evaluation request correlated to one accepted wake-up."""

    wake_up_context: WorkerWakeUpContext
    evaluated_at: datetime
    max_goals: int

    def __post_init__(self) -> None:
        if type(self.wake_up_context) is not WorkerWakeUpContext:
            raise TypeError("wake_up_context must be WorkerWakeUpContext")
        if self.wake_up_context.disposition is not WorkerWakeUpDisposition.ACCEPTED:
            raise ValueError(
                "wake_up_context.disposition must be ACCEPTED for goal evaluation"
            )
        object.__setattr__(
            self,
            "evaluated_at",
            require_aware_utc(self.evaluated_at, label="evaluated_at"),
        )
        value = require_non_negative_int(self.max_goals, label="max_goals")
        if value <= 0:
            raise ValueError("max_goals must be positive")


@dataclass(frozen=True, slots=True)
class GoalEvaluationBatchResult:
    """One bounded evaluation batch for a worker wake-up."""

    worker_instance_id: WorkerInstanceId
    wake_up_id: WakeUpId
    evaluated_at: datetime
    decisions: tuple[GoalEvaluationDecision, ...]
    goals_considered: int
    goals_evaluated: int
    goals_skipped_status: int
    goals_skipped_not_due: int
    goals_skipped_batch_limit: int

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        validate_wake_up_id(self.wake_up_id)
        object.__setattr__(
            self,
            "evaluated_at",
            require_aware_utc(self.evaluated_at, label="evaluated_at"),
        )
        object.__setattr__(
            self,
            "decisions",
            freeze_tuple(self.decisions, label="decisions"),
        )
        if self.goals_considered < 0:
            raise ValueError("goals_considered must be non-negative")
        if self.goals_evaluated < 0:
            raise ValueError("goals_evaluated must be non-negative")
        if self.goals_skipped_status < 0:
            raise ValueError("goals_skipped_status must be non-negative")
        if self.goals_skipped_not_due < 0:
            raise ValueError("goals_skipped_not_due must be non-negative")
        if self.goals_skipped_batch_limit < 0:
            raise ValueError("goals_skipped_batch_limit must be non-negative")


def goal_evaluation_sort_key(
    *,
    priority: int,
    goal_id: WorkerGoalId,
) -> tuple[int, str]:
    """Deterministic goal ordering: higher priority first, then stable goal_id."""
    return (-priority, goal_id)


def is_goal_evaluation_due(
    *,
    policy: GoalEvaluationCadencePolicy,
    evaluated_at: datetime,
    last_evaluated_at: datetime | None,
) -> bool:
    """Return whether cadence policy permits evaluation at ``evaluated_at``."""
    evaluated_at = require_aware_utc(evaluated_at, label="evaluated_at")
    if last_evaluated_at is None:
        return True
    last_evaluated_at = require_aware_utc(last_evaluated_at, label="last_evaluated_at")
    elapsed_seconds = (evaluated_at - last_evaluated_at).total_seconds()
    return elapsed_seconds >= policy.minimum_interval_seconds


def is_progress_projection_stale(
    *,
    projection: GoalProgressProjection,
    evaluated_at: datetime,
) -> bool:
    """Return whether projection freshness policy marks data stale."""
    evaluated_at = require_aware_utc(evaluated_at, label="evaluated_at")
    if projection.observed_at is None or projection.stale_after_seconds is None:
        return False
    observed_at = require_aware_utc(projection.observed_at, label="observed_at")
    elapsed_seconds = (evaluated_at - observed_at).total_seconds()
    return elapsed_seconds > projection.stale_after_seconds
