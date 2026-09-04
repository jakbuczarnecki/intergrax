# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker collaborative work bridge service (AW-4C).

Translates bounded ``GoalEvaluationDecision.ACTION_REQUIRED`` outcomes into
typed collaborative work requests submitted through ``CollaborativeWorkIntakePort``.

Does not create Nexus Tasks, dispatch Execution, grant authority, invoke LLM,
or own canonical WorkItem/Assignment semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

from intergrax.autonomous_work.collaborative_work_intake import CollaborativeWorkIntakePort
from intergrax.autonomous_work.repository import (
    ResponsibilityRepository,
    WorkerGoalRepository,
    WorkerInstanceRepository,
)
from intergrax.contracts.autonomous_work.collaborative_work_bridge import (
    CollaborativeWorkBridgeRejectionReason,
    CollaborativeWorkRequest,
    CollaborativeWorkSubmissionDisposition,
    CollaborativeWorkSubmissionResult,
    derive_collaborative_work_request_identity,
)
from intergrax.contracts.autonomous_work.goal import WorkerGoal, WorkerGoalStatus
from intergrax.contracts.autonomous_work.goal_evaluation import (
    GoalEvaluationDecision,
    GoalEvaluationDisposition,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId, validate_worker_instance_id
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.responsibility import (
    Responsibility,
    ResponsibilityStatus,
)

_INELIGIBLE_WORKER_LIFECYCLE_STATES: Final[frozenset[WorkerLifecycleState]] = frozenset(
    {
        WorkerLifecycleState.PAUSED,
        WorkerLifecycleState.QUARANTINED,
        WorkerLifecycleState.STOPPED,
        WorkerLifecycleState.PROVISIONING,
    }
)
_INELIGIBLE_GOAL_STATUSES: Final[frozenset[WorkerGoalStatus]] = frozenset(
    {
        WorkerGoalStatus.SUSPENDED,
        WorkerGoalStatus.COMPLETED,
        WorkerGoalStatus.CANCELLED,
    }
)


class WorkerCollaborativeWorkBridgeRejected(Exception):
    """Bridge rejected a non-actionable goal evaluation decision."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def derive_collaborative_work_request_title(goal: WorkerGoal) -> str:
    """Deterministic human-readable title from canonical goal metadata."""
    return goal.objective


class WorkerCollaborativeWorkBridge:
    """Bounded bridge from goal evaluation to collaborative work intake."""

    def __init__(
        self,
        *,
        worker_instance_repository: WorkerInstanceRepository,
        responsibility_repository: ResponsibilityRepository,
        worker_goal_repository: WorkerGoalRepository,
        intake_port: CollaborativeWorkIntakePort,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._worker_instance_repository = worker_instance_repository
        self._responsibility_repository = responsibility_repository
        self._worker_goal_repository = worker_goal_repository
        self._intake_port = intake_port
        self._clock = clock or _utc_now

    def submit_from_decision(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
        decision: GoalEvaluationDecision,
    ) -> CollaborativeWorkSubmissionResult:
        """Submit one collaborative work request from an ACTION_REQUIRED decision."""
        validate_worker_instance_id(worker_instance_id)
        if type(decision) is not GoalEvaluationDecision:
            raise TypeError("decision must be GoalEvaluationDecision")
        if decision.disposition is not GoalEvaluationDisposition.ACTION_REQUIRED:
            raise WorkerCollaborativeWorkBridgeRejected(
                f"collaborative work bridge requires ACTION_REQUIRED, "
                f"got {decision.disposition.value}"
            )
        if decision.reason_code is None:
            raise WorkerCollaborativeWorkBridgeRejected(
                "collaborative work bridge requires reason_code on ACTION_REQUIRED decisions"
            )

        worker = self._worker_instance_repository.get(
            worker_instance_id=worker_instance_id,
        )
        if worker is None:
            return self._rejected(
                worker_instance_id=worker_instance_id,
                decision=decision,
                reason=CollaborativeWorkBridgeRejectionReason.OWNERSHIP_MISMATCH,
            )
        if worker.lifecycle_state in _INELIGIBLE_WORKER_LIFECYCLE_STATES:
            return self._rejected(
                worker_instance_id=worker_instance_id,
                decision=decision,
                reason=CollaborativeWorkBridgeRejectionReason.WORKER_NOT_ELIGIBLE,
            )

        goal = self._worker_goal_repository.get(goal_id=decision.goal_id)
        if goal is None:
            return self._rejected(
                worker_instance_id=worker_instance_id,
                decision=decision,
                reason=CollaborativeWorkBridgeRejectionReason.OWNERSHIP_MISMATCH,
            )
        if goal.status is not WorkerGoalStatus.ACTIVE:
            return self._rejected(
                worker_instance_id=worker_instance_id,
                decision=decision,
                reason=CollaborativeWorkBridgeRejectionReason.STALE_OR_NOT_ELIGIBLE,
            )
        if goal.goal_id != decision.goal_id:
            return self._rejected(
                worker_instance_id=worker_instance_id,
                decision=decision,
                reason=CollaborativeWorkBridgeRejectionReason.STALE_OR_NOT_ELIGIBLE,
            )
        if goal.revision != decision.goal_revision:
            return self._rejected(
                worker_instance_id=worker_instance_id,
                decision=decision,
                reason=CollaborativeWorkBridgeRejectionReason.STALE_OR_NOT_ELIGIBLE,
            )

        responsibility = self._responsibility_repository.get(
            responsibility_id=goal.responsibility_id,
        )
        if responsibility is None:
            return self._rejected(
                worker_instance_id=worker_instance_id,
                decision=decision,
                reason=CollaborativeWorkBridgeRejectionReason.OWNERSHIP_MISMATCH,
            )
        if responsibility.worker_instance_id != worker_instance_id:
            return self._rejected(
                worker_instance_id=worker_instance_id,
                decision=decision,
                reason=CollaborativeWorkBridgeRejectionReason.OWNERSHIP_MISMATCH,
            )
        if responsibility.status is not ResponsibilityStatus.ACTIVE:
            return self._rejected(
                worker_instance_id=worker_instance_id,
                decision=decision,
                reason=CollaborativeWorkBridgeRejectionReason.STALE_OR_NOT_ELIGIBLE,
            )

        request = self._build_request(
            worker_instance_id=worker_instance_id,
            responsibility=responsibility,
            goal=goal,
            decision=decision,
        )
        return self._intake_port.submit(request)

    def _build_request(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
        responsibility: Responsibility,
        goal: WorkerGoal,
        decision: GoalEvaluationDecision,
    ) -> CollaborativeWorkRequest:
        request_identity = derive_collaborative_work_request_identity(
            worker_instance_id=worker_instance_id,
            goal_id=decision.goal_id,
            wake_up_id=decision.wake_up_id,
        )
        return CollaborativeWorkRequest(
            request_identity=request_identity,
            worker_instance_id=worker_instance_id,
            responsibility_id=responsibility.responsibility_id,
            goal_id=goal.goal_id,
            goal_revision=decision.goal_revision,
            wake_up_id=decision.wake_up_id,
            decision_disposition=GoalEvaluationDisposition.ACTION_REQUIRED,
            reason=decision.reason,
            reason_code=decision.reason_code,
            evidence_refs=decision.evidence_refs,
            progress_projection_ref=decision.progress_projection_ref,
            requested_priority=goal.priority,
            evaluated_at=decision.evaluated_at,
            requested_at=self._clock(),
            title=derive_collaborative_work_request_title(goal),
        )

    def _rejected(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
        decision: GoalEvaluationDecision,
        reason: CollaborativeWorkBridgeRejectionReason,
    ) -> CollaborativeWorkSubmissionResult:
        request_identity = derive_collaborative_work_request_identity(
            worker_instance_id=worker_instance_id,
            goal_id=decision.goal_id,
            wake_up_id=decision.wake_up_id,
        )
        return CollaborativeWorkSubmissionResult(
            disposition=CollaborativeWorkSubmissionDisposition.REJECTED,
            request_identity=request_identity,
            rejection_reason=reason,
        )
