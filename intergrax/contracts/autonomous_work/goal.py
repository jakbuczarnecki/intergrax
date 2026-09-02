# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""WorkerGoal semantic contract (AW-1A)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.autonomous_work._validation import (
    freeze_tuple,
    require_non_empty_text,
    require_non_negative_int,
)
from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WorkerGoalId,
    validate_responsibility_id,
    validate_worker_goal_id,
)
from intergrax.contracts.autonomous_work.references import (
    DeadlineOrCadenceRef,
    EvaluationCadenceRef,
    MetricRef,
    ProgressProjectionRef,
    SlaSloRef,
    SuccessCriteriaRef,
    validate_deadline_or_cadence_ref,
    validate_evaluation_cadence_ref,
    validate_metric_ref,
    validate_progress_projection_ref,
    validate_sla_slo_ref,
    validate_success_criteria_ref,
)
from intergrax.contracts.autonomous_work.revision import Revision, validate_revision


class WorkerGoalStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class WorkerGoal:
    """Durable measurable outcome — not a prompt, task, or authority grant."""

    goal_id: WorkerGoalId
    responsibility_id: ResponsibilityId
    objective: str
    success_criteria: SuccessCriteriaRef
    metric_refs: tuple[MetricRef, ...]
    sla_slo_refs: tuple[SlaSloRef, ...]
    deadline_or_cadence: DeadlineOrCadenceRef
    priority: int
    status: WorkerGoalStatus
    progress_projection_ref: ProgressProjectionRef
    evaluation_cadence_ref: EvaluationCadenceRef
    revision: Revision

    def __post_init__(self) -> None:
        validate_worker_goal_id(self.goal_id)
        validate_responsibility_id(self.responsibility_id)
        require_non_empty_text(self.objective, label="objective")
        validate_success_criteria_ref(self.success_criteria)
        object.__setattr__(
            self,
            "metric_refs",
            freeze_tuple(self.metric_refs, label="metric_refs"),
        )
        object.__setattr__(
            self,
            "sla_slo_refs",
            freeze_tuple(self.sla_slo_refs, label="sla_slo_refs"),
        )
        for index, metric_ref in enumerate(self.metric_refs):
            validate_metric_ref(metric_ref)
        for index, sla_ref in enumerate(self.sla_slo_refs):
            validate_sla_slo_ref(sla_ref)
        validate_deadline_or_cadence_ref(self.deadline_or_cadence)
        require_non_negative_int(self.priority, label="priority")
        if type(self.status) is not WorkerGoalStatus:
            raise TypeError("status must be WorkerGoalStatus")
        validate_progress_projection_ref(self.progress_projection_ref)
        validate_evaluation_cadence_ref(self.evaluation_cadence_ref)
        if type(self.revision) is not Revision:
            raise TypeError("revision must be Revision")
        validate_revision(self.revision)
