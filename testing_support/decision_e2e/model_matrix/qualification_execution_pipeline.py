# © Artur Czarnecki. All rights reserved.

"""Registry → planner → cohort executor (DS-E2E-15J-L1.R6-LIVE, no analysis)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from testing_support.decision_e2e.model_matrix.profiles import ModelQualificationProfile
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionResult,
    QualificationCohortExecutor,
    worst_cohort_exit_code,
)
from testing_support.decision_e2e.model_matrix.qualification_plan import (
    DEFAULT_COHORT_RUN_COUNT,
    ProfileCohortPlan,
    build_cohort_plans,
)


@dataclass(frozen=True, slots=True)
class QualificationExecutionPipelineResult:
    plans: tuple[ProfileCohortPlan, ...]
    cohort_results: tuple[CohortExecutionResult, ...]
    worst_exit_code: int


async def run_qualification_execution_pipeline(
    repo_root: Path,
    profiles: tuple[ModelQualificationProfile, ...],
    *,
    run_count: int = DEFAULT_COHORT_RUN_COUNT,
    ollama_base_url: str = "http://127.0.0.1:11434",
    env_digest: str | None = None,
    resume: bool = False,
    finalize_only: bool = False,
    executor: QualificationCohortExecutor | None = None,
) -> QualificationExecutionPipelineResult:
    plans = build_cohort_plans(
        repo_root,
        profiles,
        run_count=run_count,
        ollama_base_url=ollama_base_url,
        env_digest=env_digest,
    )
    cohort_executor = executor or QualificationCohortExecutor(
        repo_root=repo_root,
        ollama_base_url=ollama_base_url,
    )
    cohort_results = await cohort_executor.execute_plans(
        plans,
        resume=resume,
        finalize_only=finalize_only,
    )
    return QualificationExecutionPipelineResult(
        plans=plans,
        cohort_results=cohort_results,
        worst_exit_code=worst_cohort_exit_code(cohort_results),
    )


__all__ = [
    "QualificationExecutionPipelineResult",
    "run_qualification_execution_pipeline",
]
