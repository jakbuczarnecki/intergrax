# © Artur Czarnecki. All rights reserved.

"""Registry → planner → cohort executor (DS-E2E-15J-L1.R6-LIVE, no analysis)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from testing_support.decision_e2e.model_matrix.model_qualification_contract import (
    ModelQualificationContract,
    profiles_from_contracts,
)
from testing_support.decision_e2e.model_matrix.model_qualification_outcome import (
    ModelQualificationOutcome,
    outcomes_from_execution,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionResult,
    QualificationCohortExecutor,
    worst_cohort_exit_code,
)
from testing_support.decision_e2e.model_matrix.model_execution_provider import (
    ModelExecutionProvider,
)
from testing_support.decision_e2e.model_matrix.qualification_plan import (
    DEFAULT_COHORT_RUN_COUNT,
    ProfileCohortPlan,
    R6_TASK_ID,
    build_cohort_plans,
)
from testing_support.decision_e2e.model_matrix.registry import qualification_matrix_version


@dataclass(frozen=True, slots=True)
class QualificationExecutionPipelineResult:
    plans: tuple[ProfileCohortPlan, ...]
    cohort_results: tuple[CohortExecutionResult, ...]
    outcomes: tuple[ModelQualificationOutcome, ...]
    worst_exit_code: int


async def run_qualification_execution_pipeline(
    repo_root: Path,
    models: tuple[ModelQualificationContract, ...],
    *,
    run_count: int = DEFAULT_COHORT_RUN_COUNT,
    ollama_base_url: str = "http://127.0.0.1:11434",
    env_digest: str | None = None,
    resume: bool = False,
    finalize_only: bool = False,
    executor: QualificationCohortExecutor | None = None,
    execution_provider: ModelExecutionProvider | None = None,
) -> QualificationExecutionPipelineResult:
    profiles = profiles_from_contracts(models)
    plans = build_cohort_plans(
        repo_root,
        profiles,
        run_count=run_count,
        ollama_base_url=ollama_base_url,
        env_digest=env_digest,
        execution_provider=execution_provider,
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
    outcomes = outcomes_from_execution(
        plans,
        cohort_results,
        matrix_version=qualification_matrix_version(),
        qualification_task_id=R6_TASK_ID,
    )
    return QualificationExecutionPipelineResult(
        plans=plans,
        cohort_results=cohort_results,
        outcomes=outcomes,
        worst_exit_code=worst_cohort_exit_code(cohort_results),
    )


__all__ = [
    "QualificationExecutionPipelineResult",
    "run_qualification_execution_pipeline",
]
