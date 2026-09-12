# © Artur Czarnecki. All rights reserved.

"""Unified per-model qualification outcomes for R6 multi-model runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.local_ai_incident_qualification import QualificationCliExit
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationSessionState,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionResult,
    CohortExecutionStatus,
)
from testing_support.decision_e2e.model_matrix.qualification_plan import ProfileCohortPlan


@dataclass(frozen=True, slots=True)
class ModelQualificationOutcome:
    """Comparable qualification result for one model in a multi-model cohort."""

    profile_key: str
    provider: str
    model_name: str
    matrix_version: str
    qualification_task_id: str
    evaluated_at: datetime
    status: CohortExecutionStatus
    exit_code: QualificationCliExit
    session_state: QualificationSessionState | None


def outcomes_from_execution(
    plans: tuple[ProfileCohortPlan, ...],
    cohort_results: tuple[CohortExecutionResult, ...],
    *,
    matrix_version: str,
    qualification_task_id: str,
    evaluated_at: datetime | None = None,
) -> tuple[ModelQualificationOutcome, ...]:
    stamp = evaluated_at or datetime.now(tz=UTC)
    plan_by_key = {plan.profile.profile_key: plan for plan in plans}
    outcomes: list[ModelQualificationOutcome] = []
    for result in cohort_results:
        plan = plan_by_key[result.profile_key]
        outcomes.append(
            ModelQualificationOutcome(
                profile_key=result.profile_key,
                provider=plan.profile.provider,
                model_name=plan.profile.model_name,
                matrix_version=matrix_version,
                qualification_task_id=qualification_task_id,
                evaluated_at=stamp,
                status=result.status,
                exit_code=result.exit_code,
                session_state=result.session_state,
            )
        )
    return tuple(outcomes)


__all__ = ["ModelQualificationOutcome", "outcomes_from_execution"]
