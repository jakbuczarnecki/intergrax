# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded proactive WorkerGoal evaluation service (AW-4B).

Consumes accepted AW-4A wake-up context, loads canonical goals for the
target worker, applies cadence and batch limits, and returns typed decisions.

Does not create WorkItems, dispatch Execution, grant authority, or schedule
future wake-ups.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

from intergrax.autonomous_work.goal_evaluation_ports import (
    GoalEvaluationCadenceResolutionError,
    GoalEvaluationCadenceResolver,
    GoalEvaluationCadenceStateStore,
    GoalProgressProjectionResolver,
    WorkerGoalEvaluator,
    cadence_due_at,
)
from intergrax.autonomous_work.repository import (
    ResponsibilityRepository,
    WorkerGoalRepository,
)
from intergrax.contracts.autonomous_work.continuity import WorkContinuityState
from intergrax.contracts.autonomous_work.goal import WorkerGoal, WorkerGoalStatus
from intergrax.contracts.autonomous_work.goal_evaluation import (
    GoalEvaluationBatchResult,
    GoalEvaluationDecision,
    GoalEvaluationDisposition,
    GoalEvaluationReasonCode,
    WorkerGoalEvaluationRequest,
    goal_evaluation_sort_key,
)
from intergrax.contracts.autonomous_work.ids import WakeUpId, WorkerGoalId, WorkerInstanceId
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.wake_up import (
    WorkerWakeUpContext,
    WorkerWakeUpDisposition,
    WorkerWakeUpResult,
)

_INELIGIBLE_WORKER_LIFECYCLE_STATES: Final[frozenset[WorkerLifecycleState]] = frozenset(
    {
        WorkerLifecycleState.PAUSED,
        WorkerLifecycleState.QUARANTINED,
        WorkerLifecycleState.STOPPED,
        WorkerLifecycleState.PROVISIONING,
    }
)
_EVALUABLE_GOAL_STATUSES: Final[frozenset[WorkerGoalStatus]] = frozenset(
    {WorkerGoalStatus.ACTIVE}
)


class WorkerGoalEvaluationRejected(Exception):
    """Goal evaluation was requested for a non-accepted wake-up disposition."""


class WorkerGoalEvaluationService:
    """Bounded proactive goal evaluation for one accepted wake-up."""

    def __init__(
        self,
        *,
        responsibility_repository: ResponsibilityRepository,
        worker_goal_repository: WorkerGoalRepository,
        cadence_resolver: GoalEvaluationCadenceResolver,
        progress_projection_resolver: GoalProgressProjectionResolver,
        goal_evaluator: WorkerGoalEvaluator,
        cadence_state: GoalEvaluationCadenceStateStore,
    ) -> None:
        self._responsibility_repository = responsibility_repository
        self._worker_goal_repository = worker_goal_repository
        self._cadence_resolver = cadence_resolver
        self._progress_projection_resolver = progress_projection_resolver
        self._goal_evaluator = goal_evaluator
        self._cadence_state = cadence_state

    def evaluate(self, request: WorkerGoalEvaluationRequest) -> GoalEvaluationBatchResult:
        """Evaluate one bounded batch for an accepted wake-up context."""
        if type(request) is not WorkerGoalEvaluationRequest:
            raise TypeError("request must be WorkerGoalEvaluationRequest")
        context = request.wake_up_context
        if context.disposition is not WorkerWakeUpDisposition.ACCEPTED:
            raise WorkerGoalEvaluationRejected(
                f"goal evaluation requires ACCEPTED wake-up, got {context.disposition.value}"
            )
        if context.worker_instance.lifecycle_state in _INELIGIBLE_WORKER_LIFECYCLE_STATES:
            return self._empty_batch(
                context=context,
                evaluated_at=request.evaluated_at,
            )
        return self._evaluate_accepted(request=request)

    def evaluate_from_wake_up_result(
        self,
        result: WorkerWakeUpResult,
        *,
        max_goals: int,
        evaluated_at: datetime | None = None,
    ) -> GoalEvaluationBatchResult | None:
        """Evaluate only when wake-up admission was ACCEPTED."""
        if result.disposition is not WorkerWakeUpDisposition.ACCEPTED:
            return None
        if result.context is None:
            raise ValueError("ACCEPTED wake-up result requires context")
        resolved_evaluated_at = (
            evaluated_at if evaluated_at is not None else result.context.accepted_at
        )
        request = WorkerGoalEvaluationRequest(
            wake_up_context=result.context,
            evaluated_at=resolved_evaluated_at,
            max_goals=max_goals,
        )
        return self.evaluate(request)

    def _evaluate_accepted(
        self,
        *,
        request: WorkerGoalEvaluationRequest,
    ) -> GoalEvaluationBatchResult:
        context = request.wake_up_context
        worker_instance_id = context.worker_instance.worker_instance_id
        wake_up_id = context.wake_up_signal.wake_up_id
        continuity_state = context.continuity_state
        goals = self._load_worker_goals(worker_instance_id=worker_instance_id)
        goals.sort(
            key=lambda goal: goal_evaluation_sort_key(
                priority=goal.priority,
                goal_id=goal.goal_id,
            )
        )

        decisions: list[GoalEvaluationDecision] = []
        goals_considered = 0
        goals_evaluated = 0
        goals_skipped_status = 0
        goals_skipped_not_due = 0
        goals_skipped_batch_limit = 0
        evaluated_goal_ids: set[str] = set()

        for goal in goals:
            if goal.status not in _EVALUABLE_GOAL_STATUSES:
                goals_skipped_status += 1
                continue
            goals_considered += 1
            if goal.goal_id in evaluated_goal_ids:
                continue
            try:
                due = cadence_due_at(
                    resolver=self._cadence_resolver,
                    cadence_state=self._cadence_state,
                    goal=goal,
                    evaluated_at=request.evaluated_at,
                )
            except (GoalEvaluationCadenceResolutionError, KeyError):
                if goals_evaluated >= request.max_goals:
                    goals_skipped_batch_limit += 1
                    continue
                decision = GoalEvaluationDecision(
                    goal_id=goal.goal_id,
                    evaluated_at=request.evaluated_at,
                    disposition=GoalEvaluationDisposition.NOT_EVALUABLE,
                    reason_code=GoalEvaluationReasonCode.CADENCE_POLICY_UNAVAILABLE,
                    reason="evaluation cadence policy could not be resolved",
                    evidence_refs=(goal.evaluation_cadence_ref,),
                    wake_up_id=wake_up_id,
                )
                decisions.append(decision)
                evaluated_goal_ids.add(goal.goal_id)
                goals_evaluated += 1
                self._record_cadence(goal_id=goal.goal_id, evaluated_at=request.evaluated_at)
                continue
            if not due:
                goals_skipped_not_due += 1
                decisions.append(
                    GoalEvaluationDecision(
                        goal_id=goal.goal_id,
                        evaluated_at=request.evaluated_at,
                        disposition=GoalEvaluationDisposition.NOT_DUE,
                        reason="goal evaluation cadence not yet eligible",
                        wake_up_id=wake_up_id,
                    )
                )
                continue
            if goals_evaluated >= request.max_goals:
                goals_skipped_batch_limit += 1
                continue
            decision = self._evaluate_goal(
                goal=goal,
                evaluated_at=request.evaluated_at,
                continuity_state=continuity_state,
                wake_up_id=wake_up_id,
            )
            decisions.append(decision)
            evaluated_goal_ids.add(goal.goal_id)
            goals_evaluated += 1
            self._record_cadence(goal_id=goal.goal_id, evaluated_at=request.evaluated_at)

        return GoalEvaluationBatchResult(
            worker_instance_id=worker_instance_id,
            wake_up_id=wake_up_id,
            evaluated_at=request.evaluated_at,
            decisions=tuple(decisions),
            goals_considered=goals_considered,
            goals_evaluated=goals_evaluated,
            goals_skipped_status=goals_skipped_status,
            goals_skipped_not_due=goals_skipped_not_due,
            goals_skipped_batch_limit=goals_skipped_batch_limit,
        )

    def _record_cadence(
        self,
        *,
        goal_id: WorkerGoalId,
        evaluated_at: datetime,
    ) -> None:
        self._cadence_state.record_evaluated(
            goal_id=goal_id,
            evaluated_at=evaluated_at,
        )

    def _evaluate_goal(
        self,
        *,
        goal: WorkerGoal,
        evaluated_at: datetime,
        continuity_state: WorkContinuityState | None,
        wake_up_id: WakeUpId,
    ) -> GoalEvaluationDecision:
        projection = self._progress_projection_resolver.resolve(
            projection_ref=goal.progress_projection_ref,
            goal=goal,
        )
        if projection is None:
            return GoalEvaluationDecision(
                goal_id=goal.goal_id,
                evaluated_at=evaluated_at,
                disposition=GoalEvaluationDisposition.NOT_EVALUABLE,
                reason_code=GoalEvaluationReasonCode.PROGRESS_PROJECTION_UNAVAILABLE,
                reason="progress projection could not be resolved",
                evidence_refs=(goal.progress_projection_ref,),
                progress_projection_ref=goal.progress_projection_ref,
                wake_up_id=wake_up_id,
            )
        return self._goal_evaluator.evaluate(
            goal=goal,
            projection=projection,
            evaluated_at=evaluated_at,
            continuity_state=continuity_state,
            wake_up_id=wake_up_id,
        )

    def _load_worker_goals(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
    ) -> list[WorkerGoal]:
        responsibilities = self._responsibility_repository.list_for_worker_instance(
            worker_instance_id=worker_instance_id,
        )
        goals: list[WorkerGoal] = []
        for responsibility in responsibilities:
            if responsibility.worker_instance_id != worker_instance_id:
                continue
            responsibility_goals = self._worker_goal_repository.list_for_responsibility(
                responsibility_id=responsibility.responsibility_id,
            )
            for goal in responsibility_goals:
                if goal.responsibility_id != responsibility.responsibility_id:
                    continue
                goals.append(goal)
        return goals

    def _empty_batch(
        self,
        *,
        context: WorkerWakeUpContext,
        evaluated_at: datetime,
    ) -> GoalEvaluationBatchResult:
        return GoalEvaluationBatchResult(
            worker_instance_id=context.worker_instance.worker_instance_id,
            wake_up_id=context.wake_up_signal.wake_up_id,
            evaluated_at=evaluated_at,
            decisions=(),
            goals_considered=0,
            goals_evaluated=0,
            goals_skipped_status=0,
            goals_skipped_not_due=0,
            goals_skipped_batch_limit=0,
        )
