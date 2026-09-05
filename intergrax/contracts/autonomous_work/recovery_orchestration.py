# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical worker recovery orchestration contracts (AW-6B).

Executes bounded AW-6A recovery decisions as durable recovery episodes with
resume-original-work semantics. Does not classify obstacles or mint authority.
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
from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WakeUpId,
    WorkerGoalId,
    WorkerInstanceId,
    validate_responsibility_id,
    validate_wake_up_id,
    validate_worker_goal_id,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.obstacle_recovery import (
    DECISION_POLICY_VERSION,
    RecoveryStrategy,
    WorkerObstacleSourceKind,
    WorkerRecoveryDecision,
)
from intergrax.contracts.autonomous_work.references import (
    ExternalDependencyReference,
    ProblemReference,
    WorkReference,
    validate_external_dependency_reference,
    validate_problem_reference,
    validate_work_reference,
)
from intergrax.contracts.autonomous_work.revision import Revision, validate_revision
from intergrax.contracts.execution_identity import (
    ExecutionId,
    RunId,
    validate_execution_id,
    validate_run_id,
)

RECOVERY_EPISODE_CONTRACT_VERSION: str = "aw-6b.v1"


class RecoveryEpisodeStatus(StrEnum):
    """Minimal durable recovery episode lifecycle."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    WAITING = "WAITING"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ESCALATED = "ESCALATED"
    QUARANTINED = "QUARANTINED"
    STOPPED = "STOPPED"


_TERMINAL_EPISODE_STATUSES = frozenset(
    {
        RecoveryEpisodeStatus.SUCCEEDED,
        RecoveryEpisodeStatus.FAILED,
        RecoveryEpisodeStatus.ESCALATED,
        RecoveryEpisodeStatus.QUARANTINED,
        RecoveryEpisodeStatus.STOPPED,
    }
)


class WorkerRecoveryResumeTargetKind(StrEnum):
    """Implemented original-work resume target categories only."""

    GOAL_DECISION = "goal_decision"
    OPERATOR = "operator"
    COLLABORATIVE_WORK = "collaborative_work"
    EXECUTION_SOURCE = "execution_source"


class WorkerRecoveryAttemptDisposition(StrEnum):
    """Typed recovery attempt outcome."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    WAITING = "WAITING"
    ESCALATED = "ESCALATED"
    REJECTED = "REJECTED"
    UNAVAILABLE = "UNAVAILABLE"


class WorkerRecoveryOrchestrationDisposition(StrEnum):
    """Typed orchestration service outcome."""

    RESUMED = "RESUMED"
    ATTEMPT_DISPATCHED = "ATTEMPT_DISPATCHED"
    WAITING = "WAITING"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    ESCALATED = "ESCALATED"
    QUARANTINED = "QUARANTINED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    CONFLICT = "CONFLICT"
    STALE_SOURCE = "STALE_SOURCE"
    STALE_CONTINUITY = "STALE_CONTINUITY"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    ALREADY_SUCCEEDED = "ALREADY_SUCCEEDED"
    ALREADY_TERMINAL = "ALREADY_TERMINAL"


class WorkerOriginalWorkResumeDisposition(StrEnum):
    """Typed resume-original-work preparation outcome."""

    RESUMED = "RESUMED"
    CONFLICT = "CONFLICT"
    UNAVAILABLE = "UNAVAILABLE"


def derive_recovery_episode_id(
    *,
    worker_instance_id: WorkerInstanceId,
    obstacle_id: str,
    recovery_decision_id: str,
) -> str:
    """Stable logical recovery episode identity — not a random per-invocation UUID."""

    validate_worker_instance_id(worker_instance_id)
    obstacle = require_non_empty_text(obstacle_id, label="obstacle_id")
    decision = require_non_empty_text(recovery_decision_id, label="recovery_decision_id")
    return f"{worker_instance_id}:{obstacle}:{decision}"


def derive_recovery_attempt_id(
    *,
    recovery_episode_id: str,
    attempt_number: int,
) -> str:
    episode_id = require_non_empty_text(recovery_episode_id, label="recovery_episode_id")
    if attempt_number < 1:
        raise ValueError("attempt_number must be >= 1")
    return f"{episode_id}:attempt:{attempt_number}"


def is_terminal_recovery_episode_status(status: RecoveryEpisodeStatus) -> bool:
    return status in _TERMINAL_EPISODE_STATUSES


@dataclass(frozen=True, slots=True)
class WorkerOriginalWorkSource:
    """Canonical correlation to the original work lineage."""

    worker_instance_id: WorkerInstanceId
    source_kind: WorkerObstacleSourceKind
    source_ref: str

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        if type(self.source_kind) is not WorkerObstacleSourceKind:
            raise TypeError("source_kind must be WorkerObstacleSourceKind")
        object.__setattr__(
            self,
            "source_ref",
            require_non_empty_text(self.source_ref, label="source_ref"),
        )


@dataclass(frozen=True, slots=True)
class WorkerRecoveryResumeTarget:
    """Typed resume target — references only, no executable payload."""

    kind: WorkerRecoveryResumeTargetKind
    source_ref: str
    goal_id: WorkerGoalId | None = None
    goal_revision: Revision | None = None
    responsibility_id: ResponsibilityId | None = None
    wake_up_id: WakeUpId | None = None
    collaborative_work_ref: WorkReference | None = None
    execution_id: ExecutionId | None = None
    run_id: RunId | None = None
    requested_scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not WorkerRecoveryResumeTargetKind:
            raise TypeError("kind must be WorkerRecoveryResumeTargetKind")
        object.__setattr__(
            self,
            "source_ref",
            require_non_empty_text(self.source_ref, label="source_ref"),
        )
        object.__setattr__(
            self,
            "requested_scopes",
            freeze_tuple(self.requested_scopes, label="requested_scopes"),
        )
        if self.goal_id is not None:
            validate_worker_goal_id(self.goal_id)
        if self.goal_revision is not None:
            validate_revision(self.goal_revision)
        if self.responsibility_id is not None:
            validate_responsibility_id(self.responsibility_id)
        if self.wake_up_id is not None:
            validate_wake_up_id(self.wake_up_id)
        if self.collaborative_work_ref is not None:
            validate_work_reference(self.collaborative_work_ref)
        if self.execution_id is not None:
            validate_execution_id(self.execution_id)
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.kind is WorkerRecoveryResumeTargetKind.GOAL_DECISION:
            if self.goal_id is None or self.goal_revision is None:
                raise ValueError("GOAL_DECISION resume target requires goal_id and goal_revision")
            if self.responsibility_id is None or self.wake_up_id is None:
                raise ValueError(
                    "GOAL_DECISION resume target requires responsibility_id and wake_up_id"
                )
        if self.kind is WorkerRecoveryResumeTargetKind.COLLABORATIVE_WORK:
            if self.collaborative_work_ref is None:
                raise ValueError("COLLABORATIVE_WORK resume target requires collaborative_work_ref")


@dataclass(frozen=True, slots=True)
class RecoveryExecutionBounds:
    """Typed recovery time budget — no hardcoded global limits."""

    max_elapsed_seconds: int | None = None
    deadline: datetime | None = None

    def __post_init__(self) -> None:
        if self.max_elapsed_seconds is not None and self.max_elapsed_seconds < 0:
            raise ValueError("max_elapsed_seconds must be non-negative")
        if self.deadline is not None:
            object.__setattr__(
                self,
                "deadline",
                require_aware_utc(self.deadline, label="deadline"),
            )


@dataclass(frozen=True, slots=True)
class WorkerRecoveryEpisode:
    """Immutable durable recovery episode aggregate."""

    recovery_episode_id: str
    worker_instance_id: WorkerInstanceId
    obstacle_id: str
    recovery_decision_id: str
    decision_policy_version: str
    strategy: RecoveryStrategy
    original_source: WorkerOriginalWorkSource
    resume_target: WorkerRecoveryResumeTarget
    started_at: datetime
    status: RecoveryEpisodeStatus
    attempt_count: int
    revision: Revision
    max_attempts: int | None = None
    last_attempt_at: datetime | None = None
    next_retry_at: datetime | None = None
    last_execution_id: ExecutionId | None = None
    last_failure_ref: str | None = None
    terminal_reason: str | None = None
    completed_at: datetime | None = None
    pre_recovery_lifecycle_state: WorkerLifecycleState | None = None
    dependency_ref: ExternalDependencyReference | None = None
    human_decision_ref: str | None = None
    claimed_attempt_number: int | None = None
    continuity_resume_completed: bool = False
    continuity_resume_revision: Revision | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "recovery_episode_id",
            require_non_empty_text(self.recovery_episode_id, label="recovery_episode_id"),
        )
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(
            self,
            "obstacle_id",
            require_non_empty_text(self.obstacle_id, label="obstacle_id"),
        )
        object.__setattr__(
            self,
            "recovery_decision_id",
            require_non_empty_text(self.recovery_decision_id, label="recovery_decision_id"),
        )
        object.__setattr__(
            self,
            "decision_policy_version",
            require_non_empty_text(
                self.decision_policy_version,
                label="decision_policy_version",
            ),
        )
        if type(self.strategy) is not RecoveryStrategy:
            raise TypeError("strategy must be RecoveryStrategy")
        if type(self.original_source) is not WorkerOriginalWorkSource:
            raise TypeError("original_source must be WorkerOriginalWorkSource")
        if type(self.resume_target) is not WorkerRecoveryResumeTarget:
            raise TypeError("resume_target must be WorkerRecoveryResumeTarget")
        object.__setattr__(
            self,
            "started_at",
            require_aware_utc(self.started_at, label="started_at"),
        )
        if type(self.status) is not RecoveryEpisodeStatus:
            raise TypeError("status must be RecoveryEpisodeStatus")
        if self.attempt_count < 0:
            raise ValueError("attempt_count must be non-negative")
        validate_revision(self.revision)
        if self.max_attempts is not None and self.max_attempts < 0:
            raise ValueError("max_attempts must be non-negative")
        if self.last_attempt_at is not None:
            object.__setattr__(
                self,
                "last_attempt_at",
                require_aware_utc(self.last_attempt_at, label="last_attempt_at"),
            )
        if self.next_retry_at is not None:
            object.__setattr__(
                self,
                "next_retry_at",
                require_aware_utc(self.next_retry_at, label="next_retry_at"),
            )
        if self.last_execution_id is not None:
            validate_execution_id(self.last_execution_id)
        if self.last_failure_ref is not None:
            object.__setattr__(
                self,
                "last_failure_ref",
                require_non_empty_text(self.last_failure_ref, label="last_failure_ref"),
            )
        if self.completed_at is not None:
            object.__setattr__(
                self,
                "completed_at",
                require_aware_utc(self.completed_at, label="completed_at"),
            )
        if self.pre_recovery_lifecycle_state is not None:
            if type(self.pre_recovery_lifecycle_state) is not WorkerLifecycleState:
                raise TypeError("pre_recovery_lifecycle_state must be WorkerLifecycleState")
        if self.dependency_ref is not None:
            validate_external_dependency_reference(self.dependency_ref)
        if self.human_decision_ref is not None:
            object.__setattr__(
                self,
                "human_decision_ref",
                require_non_empty_text(self.human_decision_ref, label="human_decision_ref"),
            )
        if self.claimed_attempt_number is not None and self.claimed_attempt_number < 1:
            raise ValueError("claimed_attempt_number must be >= 1 when provided")
        if self.continuity_resume_revision is not None:
            validate_revision(self.continuity_resume_revision)
        if self.continuity_resume_completed and self.continuity_resume_revision is None:
            raise ValueError(
                "continuity_resume_completed requires continuity_resume_revision"
            )
        if not self.continuity_resume_completed and self.continuity_resume_revision is not None:
            raise ValueError(
                "continuity_resume_revision requires continuity_resume_completed"
            )
        _validate_episode_invariants(self)


def _validate_episode_invariants(episode: WorkerRecoveryEpisode) -> None:
    if is_terminal_recovery_episode_status(episode.status):
        if episode.next_retry_at is not None:
            raise ValueError("terminal episode must not carry next_retry_at")
        if episode.claimed_attempt_number is not None:
            raise ValueError("terminal episode must not carry claimed_attempt_number")
    if episode.status is RecoveryEpisodeStatus.WAITING and episode.next_retry_at is None:
        if episode.dependency_ref is None:
            raise ValueError("WAITING episode requires next_retry_at or dependency_ref")
    if episode.status is RecoveryEpisodeStatus.WAITING_FOR_HUMAN and episode.human_decision_ref is None:
        raise ValueError("WAITING_FOR_HUMAN episode requires human_decision_ref")
    if episode.status is RecoveryEpisodeStatus.SUCCEEDED and episode.terminal_reason == "FAILED":
        raise ValueError("SUCCEEDED episode cannot carry terminal_reason=FAILED")


@dataclass(frozen=True, slots=True)
class WorkerRecoveryAttemptResult:
    """Immutable recovery attempt evidence."""

    recovery_episode_id: str
    attempt_number: int
    attempt_id: str
    strategy: RecoveryStrategy
    disposition: WorkerRecoveryAttemptDisposition
    started_at: datetime
    finished_at: datetime
    execution_id: ExecutionId | None = None
    failure_ref: str | None = None
    next_retry_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "recovery_episode_id",
            require_non_empty_text(self.recovery_episode_id, label="recovery_episode_id"),
        )
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be >= 1")
        object.__setattr__(
            self,
            "attempt_id",
            require_non_empty_text(self.attempt_id, label="attempt_id"),
        )
        if type(self.strategy) is not RecoveryStrategy:
            raise TypeError("strategy must be RecoveryStrategy")
        if type(self.disposition) is not WorkerRecoveryAttemptDisposition:
            raise TypeError("disposition must be WorkerRecoveryAttemptDisposition")
        object.__setattr__(
            self,
            "started_at",
            require_aware_utc(self.started_at, label="started_at"),
        )
        object.__setattr__(
            self,
            "finished_at",
            require_aware_utc(self.finished_at, label="finished_at"),
        )
        if self.execution_id is not None:
            validate_execution_id(self.execution_id)
        if self.failure_ref is not None:
            object.__setattr__(
                self,
                "failure_ref",
                require_non_empty_text(self.failure_ref, label="failure_ref"),
            )
        if self.next_retry_at is not None:
            object.__setattr__(
                self,
                "next_retry_at",
                require_aware_utc(self.next_retry_at, label="next_retry_at"),
            )


@dataclass(frozen=True, slots=True)
class WorkerOriginalWorkResumeIntent:
    """Typed resume-original-work intent — no executable callable."""

    worker_instance_id: WorkerInstanceId
    recovery_episode_id: str
    original_source: WorkerOriginalWorkSource
    resume_target: WorkerRecoveryResumeTarget
    continuity_revision: Revision
    created_at: datetime

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(
            self,
            "recovery_episode_id",
            require_non_empty_text(self.recovery_episode_id, label="recovery_episode_id"),
        )
        if type(self.original_source) is not WorkerOriginalWorkSource:
            raise TypeError("original_source must be WorkerOriginalWorkSource")
        if type(self.resume_target) is not WorkerRecoveryResumeTarget:
            raise TypeError("resume_target must be WorkerRecoveryResumeTarget")
        validate_revision(self.continuity_revision)
        object.__setattr__(
            self,
            "created_at",
            require_aware_utc(self.created_at, label="created_at"),
        )


@dataclass(frozen=True, slots=True)
class WorkerOriginalWorkResumeResult:
    """Typed resume-original-work outcome — no silent stale continuity success."""

    disposition: WorkerOriginalWorkResumeDisposition
    resume_intent: WorkerOriginalWorkResumeIntent | None = None
    continuity_revision: Revision | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not WorkerOriginalWorkResumeDisposition:
            raise TypeError("disposition must be WorkerOriginalWorkResumeDisposition")
        if self.resume_intent is not None:
            if type(self.resume_intent) is not WorkerOriginalWorkResumeIntent:
                raise TypeError("resume_intent must be WorkerOriginalWorkResumeIntent")
        if self.continuity_revision is not None:
            validate_revision(self.continuity_revision)
        if self.disposition is WorkerOriginalWorkResumeDisposition.RESUMED:
            if self.resume_intent is None:
                raise ValueError("RESUMED disposition requires resume_intent")
            if self.continuity_revision is None:
                raise ValueError("RESUMED disposition requires continuity_revision")


@dataclass(frozen=True, slots=True)
class WorkerRecoveryOrchestrationRequest:
    """Execute one bounded recovery orchestration step for a decision."""

    decision: WorkerRecoveryDecision
    original_source: WorkerOriginalWorkSource
    resume_target: WorkerRecoveryResumeTarget
    bounds: RecoveryExecutionBounds | None = None
    continuity_expected_revision: Revision | None = None
    pre_recovery_lifecycle_state: WorkerLifecycleState | None = None
    evidence_refs: tuple[ProblemReference, ...] = ()

    def __post_init__(self) -> None:
        if type(self.decision) is not WorkerRecoveryDecision:
            raise TypeError("decision must be WorkerRecoveryDecision")
        if type(self.original_source) is not WorkerOriginalWorkSource:
            raise TypeError("original_source must be WorkerOriginalWorkSource")
        if type(self.resume_target) is not WorkerRecoveryResumeTarget:
            raise TypeError("resume_target must be WorkerRecoveryResumeTarget")
        if self.bounds is not None and type(self.bounds) is not RecoveryExecutionBounds:
            raise TypeError("bounds must be RecoveryExecutionBounds")
        if self.continuity_expected_revision is not None:
            validate_revision(self.continuity_expected_revision)
        if self.pre_recovery_lifecycle_state is not None:
            if type(self.pre_recovery_lifecycle_state) is not WorkerLifecycleState:
                raise TypeError("pre_recovery_lifecycle_state must be WorkerLifecycleState")
        object.__setattr__(
            self,
            "evidence_refs",
            freeze_tuple(self.evidence_refs, label="evidence_refs"),
        )
        for ref in self.evidence_refs:
            validate_problem_reference(ref)


@dataclass(frozen=True, slots=True)
class WorkerRecoveryOrchestrationResult:
    """Typed orchestration outcome with durable episode state."""

    disposition: WorkerRecoveryOrchestrationDisposition
    episode: WorkerRecoveryEpisode
    attempt_result: WorkerRecoveryAttemptResult | None = None
    resume_intent: WorkerOriginalWorkResumeIntent | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not WorkerRecoveryOrchestrationDisposition:
            raise TypeError("disposition must be WorkerRecoveryOrchestrationDisposition")
        if type(self.episode) is not WorkerRecoveryEpisode:
            raise TypeError("episode must be WorkerRecoveryEpisode")
        if self.attempt_result is not None:
            if type(self.attempt_result) is not WorkerRecoveryAttemptResult:
                raise TypeError("attempt_result must be WorkerRecoveryAttemptResult")
        if self.resume_intent is not None:
            if type(self.resume_intent) is not WorkerOriginalWorkResumeIntent:
                raise TypeError("resume_intent must be WorkerOriginalWorkResumeIntent")
        if self.disposition is WorkerRecoveryOrchestrationDisposition.RESUMED:
            if self.resume_intent is None:
                raise ValueError("RESUMED disposition requires resume_intent")


def recovery_episodes_logically_equivalent(
    left: WorkerRecoveryEpisode,
    right: WorkerRecoveryEpisode,
) -> bool:
    """Return whether two episodes represent the same durable semantic content."""
    return (
        left.recovery_episode_id == right.recovery_episode_id
        and left.worker_instance_id == right.worker_instance_id
        and left.obstacle_id == right.obstacle_id
        and left.recovery_decision_id == right.recovery_decision_id
        and left.decision_policy_version == right.decision_policy_version
        and left.strategy is right.strategy
        and left.original_source == right.original_source
        and left.resume_target == right.resume_target
        and left.max_attempts == right.max_attempts
        and left.pre_recovery_lifecycle_state == right.pre_recovery_lifecycle_state
        and left.dependency_ref == right.dependency_ref
        and left.human_decision_ref == right.human_decision_ref
    )


def default_decision_policy_version() -> str:
    return DECISION_POLICY_VERSION
