# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Collaborative Work intake bridge contracts (AW-4C).

Typed request/result boundary between Autonomous Work goal evaluation and
Collaborative Work canonical WorkItem intake (MP-2). Autonomous Work does not
own business WorkItem or Assignment semantics.
"""

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
from intergrax.contracts.autonomous_work.goal_evaluation import (
    GoalEvaluationDisposition,
    GoalEvaluationReasonCode,
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
from intergrax.contracts.autonomous_work.references import (
    ProgressProjectionRef,
    WorkReference,
    validate_progress_projection_ref,
    validate_work_reference,
)
from intergrax.contracts.autonomous_work.revision import Revision, validate_revision


class CollaborativeWorkSubmissionDisposition(StrEnum):
    """Typed outcome from Collaborative Work intake."""

    ACCEPTED = "ACCEPTED"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    UNAVAILABLE = "UNAVAILABLE"
    REJECTED = "REJECTED"


class CollaborativeWorkBridgeRejectionReason(StrEnum):
    """Bridge-level rejection before or during intake eligibility checks."""

    INVALID_DECISION_DISPOSITION = "INVALID_DECISION_DISPOSITION"
    STALE_OR_NOT_ELIGIBLE = "STALE_OR_NOT_ELIGIBLE"
    OWNERSHIP_MISMATCH = "OWNERSHIP_MISMATCH"
    WORKER_NOT_ELIGIBLE = "WORKER_NOT_ELIGIBLE"


@dataclass(frozen=True, slots=True)
class CollaborativeWorkRequestIdentity:
    """Stable logical identity for idempotent collaborative work intake."""

    worker_instance_id: WorkerInstanceId
    goal_id: WorkerGoalId
    wake_up_id: WakeUpId

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        validate_worker_goal_id(self.goal_id)
        validate_wake_up_id(self.wake_up_id)

    @property
    def identity_key(self) -> str:
        """Deterministic idempotency key — not transport delivery identity."""
        return (
            f"{self.worker_instance_id}:{self.goal_id}:{self.wake_up_id}"
        )


def derive_collaborative_work_request_identity(
    *,
    worker_instance_id: WorkerInstanceId,
    goal_id: WorkerGoalId,
    wake_up_id: WakeUpId,
) -> CollaborativeWorkRequestIdentity:
    """Derive stable request identity from wake/evaluation correlation."""
    return CollaborativeWorkRequestIdentity(
        worker_instance_id=worker_instance_id,
        goal_id=goal_id,
        wake_up_id=wake_up_id,
    )


@dataclass(frozen=True, slots=True)
class CollaborativeWorkRequest:
    """Evidence that Autonomous Work requires canonical collaborative work."""

    request_identity: CollaborativeWorkRequestIdentity
    worker_instance_id: WorkerInstanceId
    responsibility_id: ResponsibilityId
    goal_id: WorkerGoalId
    goal_revision: Revision
    wake_up_id: WakeUpId
    decision_disposition: GoalEvaluationDisposition
    reason: str
    evaluated_at: datetime
    requested_at: datetime
    requested_priority: int
    title: str
    reason_code: GoalEvaluationReasonCode | None = None
    evidence_refs: tuple[str, ...] = ()
    progress_projection_ref: ProgressProjectionRef | None = None

    def __post_init__(self) -> None:
        if type(self.request_identity) is not CollaborativeWorkRequestIdentity:
            raise TypeError("request_identity must be CollaborativeWorkRequestIdentity")
        validate_worker_instance_id(self.worker_instance_id)
        validate_responsibility_id(self.responsibility_id)
        validate_worker_goal_id(self.goal_id)
        validate_revision(self.goal_revision)
        validate_wake_up_id(self.wake_up_id)
        if self.decision_disposition is not GoalEvaluationDisposition.ACTION_REQUIRED:
            raise ValueError(
                "decision_disposition must be ACTION_REQUIRED for collaborative work requests"
            )
        object.__setattr__(
            self,
            "reason",
            require_non_empty_text(self.reason, label="reason"),
        )
        object.__setattr__(
            self,
            "evaluated_at",
            require_aware_utc(self.evaluated_at, label="evaluated_at"),
        )
        object.__setattr__(
            self,
            "requested_at",
            require_aware_utc(self.requested_at, label="requested_at"),
        )
        require_non_negative_int(self.requested_priority, label="requested_priority")
        object.__setattr__(
            self,
            "title",
            require_non_empty_text(self.title, label="title"),
        )
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
class CollaborativeWorkSubmissionResult:
    """Typed intake outcome — does not mint fake canonical WorkItem IDs."""

    disposition: CollaborativeWorkSubmissionDisposition
    request_identity: CollaborativeWorkRequestIdentity
    collaborative_work_ref: WorkReference | None = None
    rejection_reason: CollaborativeWorkBridgeRejectionReason | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not CollaborativeWorkSubmissionDisposition:
            raise TypeError("disposition must be CollaborativeWorkSubmissionDisposition")
        if type(self.request_identity) is not CollaborativeWorkRequestIdentity:
            raise TypeError("request_identity must be CollaborativeWorkRequestIdentity")
        if self.collaborative_work_ref is not None:
            validate_work_reference(self.collaborative_work_ref)
        if self.rejection_reason is not None:
            if type(self.rejection_reason) is not CollaborativeWorkBridgeRejectionReason:
                raise TypeError(
                    "rejection_reason must be CollaborativeWorkBridgeRejectionReason"
                )
