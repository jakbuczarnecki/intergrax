# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Goal evaluation resolver/evaluator ports and reference adapters (AW-4B)."""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.continuity import WorkContinuityState
from intergrax.contracts.autonomous_work.goal import WorkerGoal
from intergrax.contracts.autonomous_work.goal_evaluation import (
    GoalEvaluationCadencePolicy,
    GoalEvaluationDecision,
    GoalEvaluationDisposition,
    GoalEvaluationReasonCode,
    GoalProgressProjection,
    is_goal_evaluation_due,
    is_progress_projection_stale,
)
from intergrax.contracts.autonomous_work.ids import WakeUpId, WorkerGoalId
from intergrax.contracts.autonomous_work.references import (
    EvaluationCadenceRef,
    ProgressProjectionRef,
)

_CADENCE_SUFFIX_PATTERN = re.compile(r"^(\d+)([smhd])$")


def parse_cadence_interval_seconds(ref: EvaluationCadenceRef) -> int | None:
    """Parse ``cadence/goal-eval-5m`` style refs into interval seconds."""
    suffix = ref.rsplit("-", 1)[-1]
    match = _CADENCE_SUFFIX_PATTERN.fullmatch(suffix)
    if match is None:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return amount * multiplier


@runtime_checkable
class GoalEvaluationCadenceResolver(Protocol):
    """Resolve ``EvaluationCadenceRef`` into a typed cadence policy."""

    def resolve(
        self,
        *,
        cadence_ref: EvaluationCadenceRef,
        goal: WorkerGoal,
    ) -> GoalEvaluationCadencePolicy: ...


@runtime_checkable
class GoalEvaluationCadenceStateReader(Protocol):
    """Read-only last-evaluation timestamps for cadence eligibility."""

    def last_evaluated_at(self, *, goal_id: WorkerGoalId) -> datetime | None: ...


@runtime_checkable
class GoalEvaluationCadenceStateRecorder(Protocol):
    """Record evaluation timestamps after a bounded batch completes."""

    def record_evaluated(
        self,
        *,
        goal_id: WorkerGoalId,
        evaluated_at: datetime,
    ) -> None: ...


@runtime_checkable
class GoalProgressProjectionResolver(Protocol):
    """Resolve ``ProgressProjectionRef`` into typed progress inputs."""

    def resolve(
        self,
        *,
        projection_ref: ProgressProjectionRef,
        goal: WorkerGoal,
    ) -> GoalProgressProjection | None: ...


@runtime_checkable
class WorkerGoalEvaluator(Protocol):
    """Semantic evaluator seam — answers whether action is required."""

    def evaluate(
        self,
        *,
        goal: WorkerGoal,
        projection: GoalProgressProjection,
        evaluated_at: datetime,
        continuity_state: WorkContinuityState | None,
        wake_up_id: WakeUpId,
    ) -> GoalEvaluationDecision: ...


class MappingGoalEvaluationCadenceResolver:
    """Reference cadence resolver backed by explicit ref→interval mappings."""

    def __init__(
        self,
        intervals: Mapping[EvaluationCadenceRef, int],
        *,
        default_interval_seconds: int | None = None,
    ) -> None:
        if default_interval_seconds is not None and default_interval_seconds <= 0:
            raise ValueError("default_interval_seconds must be positive when provided")
        self._intervals = dict(intervals)
        self._default_interval_seconds = default_interval_seconds

    def resolve(
        self,
        *,
        cadence_ref: EvaluationCadenceRef,
        goal: WorkerGoal,
    ) -> GoalEvaluationCadencePolicy:
        del goal
        interval = self._intervals.get(cadence_ref)
        if interval is None:
            interval = parse_cadence_interval_seconds(cadence_ref)
        if interval is None:
            interval = self._default_interval_seconds
        if interval is None or interval <= 0:
            raise KeyError(f"no cadence policy for {cadence_ref}")
        return GoalEvaluationCadencePolicy(minimum_interval_seconds=interval)


class InMemoryGoalEvaluationCadenceStateStore:
    """Process-local cadence state for reference deployments and tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._last_evaluated_at: dict[WorkerGoalId, datetime] = {}

    def last_evaluated_at(self, *, goal_id: WorkerGoalId) -> datetime | None:
        with self._lock:
            return self._last_evaluated_at.get(goal_id)

    def record_evaluated(
        self,
        *,
        goal_id: WorkerGoalId,
        evaluated_at: datetime,
    ) -> None:
        with self._lock:
            self._last_evaluated_at[goal_id] = evaluated_at


class MappingGoalProgressProjectionResolver:
    """Reference progress resolver backed by explicit projection mappings."""

    def __init__(
        self,
        projections: Mapping[ProgressProjectionRef, GoalProgressProjection],
    ) -> None:
        self._projections = dict(projections)

    def resolve(
        self,
        *,
        projection_ref: ProgressProjectionRef,
        goal: WorkerGoal,
    ) -> GoalProgressProjection | None:
        del goal
        return self._projections.get(projection_ref)


class DeterministicThresholdGoalEvaluator:
    """Deterministic evaluator for numeric/status threshold semantics."""

    def evaluate(
        self,
        *,
        goal: WorkerGoal,
        projection: GoalProgressProjection,
        evaluated_at: datetime,
        continuity_state: WorkContinuityState | None,
        wake_up_id: WakeUpId,
    ) -> GoalEvaluationDecision:
        if is_progress_projection_stale(projection=projection, evaluated_at=evaluated_at):
            return GoalEvaluationDecision(
                goal_id=goal.goal_id,
                evaluated_at=evaluated_at,
                disposition=GoalEvaluationDisposition.NOT_EVALUABLE,
                reason_code=GoalEvaluationReasonCode.PROGRESS_PROJECTION_STALE,
                reason="progress projection is stale for evaluation",
                evidence_refs=projection.evidence_refs,
                progress_projection_ref=projection.projection_ref,
                wake_up_id=wake_up_id,
            )

        if continuity_state is not None:
            if goal.goal_id in continuity_state.active_goal_refs:
                if continuity_state.open_work_refs:
                    return GoalEvaluationDecision(
                        goal_id=goal.goal_id,
                        evaluated_at=evaluated_at,
                        disposition=GoalEvaluationDisposition.NO_ACTION,
                        reason_code=GoalEvaluationReasonCode.OPEN_WORK_ALREADY_PENDING,
                        reason="open work already pending for active goal",
                        evidence_refs=continuity_state.open_work_refs,
                        progress_projection_ref=projection.projection_ref,
                        wake_up_id=wake_up_id,
                    )

        status = (projection.status or "").strip().lower()
        if status in {"breached", "at_risk"}:
            reason_code = (
                GoalEvaluationReasonCode.SLA_RISK
                if status == "at_risk"
                else GoalEvaluationReasonCode.THRESHOLD_BREACH
            )
            return GoalEvaluationDecision(
                goal_id=goal.goal_id,
                evaluated_at=evaluated_at,
                disposition=GoalEvaluationDisposition.ACTION_REQUIRED,
                reason_code=reason_code,
                reason=f"progress status is {status}",
                evidence_refs=projection.evidence_refs,
                progress_projection_ref=projection.projection_ref,
                wake_up_id=wake_up_id,
            )

        if status == "stalled":
            return GoalEvaluationDecision(
                goal_id=goal.goal_id,
                evaluated_at=evaluated_at,
                disposition=GoalEvaluationDisposition.ACTION_REQUIRED,
                reason_code=GoalEvaluationReasonCode.PROGRESS_STALLED,
                reason="progress is stalled",
                evidence_refs=projection.evidence_refs,
                progress_projection_ref=projection.projection_ref,
                wake_up_id=wake_up_id,
            )

        if (
            projection.current_value is not None
            and projection.target_value is not None
            and projection.current_value < projection.target_value
        ):
            return GoalEvaluationDecision(
                goal_id=goal.goal_id,
                evaluated_at=evaluated_at,
                disposition=GoalEvaluationDisposition.ACTION_REQUIRED,
                reason_code=GoalEvaluationReasonCode.SUCCESS_CRITERIA_NOT_MET,
                reason="success criteria threshold not met",
                evidence_refs=projection.evidence_refs,
                progress_projection_ref=projection.projection_ref,
                wake_up_id=wake_up_id,
            )

        return GoalEvaluationDecision(
            goal_id=goal.goal_id,
            evaluated_at=evaluated_at,
            disposition=GoalEvaluationDisposition.NO_ACTION,
            reason_code=GoalEvaluationReasonCode.CRITERIA_MET,
            reason="goal progress meets configured success criteria",
            evidence_refs=projection.evidence_refs,
            progress_projection_ref=projection.projection_ref,
            wake_up_id=wake_up_id,
        )


def cadence_due_at(
    *,
    resolver: GoalEvaluationCadenceResolver,
    cadence_state: GoalEvaluationCadenceStateReader,
    goal: WorkerGoal,
    evaluated_at: datetime,
) -> bool:
    """Pure helper composing cadence resolver and last-evaluated state."""
    policy = resolver.resolve(
        cadence_ref=goal.evaluation_cadence_ref,
        goal=goal,
    )
    last_evaluated_at = cadence_state.last_evaluated_at(goal_id=goal.goal_id)
    return is_goal_evaluation_due(
        policy=policy,
        evaluated_at=evaluated_at,
        last_evaluated_at=last_evaluated_at,
    )
