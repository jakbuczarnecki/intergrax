# © Artur Czarnecki. All rights reserved.

"""Cohort execution for multi-model qualification (DS-E2E-15J-L1.R6-LIVE)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Awaitable, Protocol

from testing_support.decision_e2e.env_bootstrap import bootstrap_qualification_environment
from testing_support.decision_e2e.local_ai_incident_qualification import (
    AiIncidentSingleRunExecutor,
    LocalQualificationOrchestrationResult,
    OllamaProviderIdentityProbe,
    QualificationCliExit,
    QualificationRunExecutor,
    resolve_repository_head_sha,
    run_local_ai_incident_qualification,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationSessionState,
)
from testing_support.decision_e2e.local_qualification_session.ollama_probe import (
    OllamaProbeConfig,
)
from testing_support.decision_e2e.model_matrix.availability import ModelAvailability
from testing_support.decision_e2e.model_matrix.qualification_cohort_failure import (
    CohortFailureAttribution,
    attribution_for_exception,
    resolve_cohort_failure,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_resume import (
    CohortResumeAction,
    decide_cohort_resume,
)
from testing_support.decision_e2e.model_matrix.qualification_plan import (
    ProfileCohortPlan,
    R6_TASK_ID,
    build_qualification_spec_for_profile,
)

QualificationPlan = ProfileCohortPlan

R6_LIVE_TASK_ID = "DS-E2E-15J-L1.R6-LIVE"


class CohortExecutionStatus(StrEnum):
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    BLOCKED_PRECONDITION = "BLOCKED_PRECONDITION"
    EXECUTED = "EXECUTED"
    ISOLATED_FAILURE = "ISOLATED_FAILURE"


@dataclass(frozen=True, slots=True)
class CohortExecutionResult:
    profile_key: str
    availability: ModelAvailability
    status: CohortExecutionStatus
    exit_code: QualificationCliExit
    session_state: QualificationSessionState | None = None
    failure: CohortFailureAttribution | None = None


QualificationCohortResult = CohortExecutionResult


class _QualificationSessionRunner(Protocol):
    def __call__(
        self,
        *,
        repo_root: Path,
        session_dir: Path,
        spec,
        config_fingerprint: str,
        source_fingerprint: str,
        provider_probe,
        run_executor: QualificationRunExecutor,
        resume: bool,
        finalize_only: bool,
        repository_head_sha: str | None,
        task_id: str,
        temperature: float,
    ) -> Awaitable[LocalQualificationOrchestrationResult]:
        ...


def _apply_profile_runtime_env(profile_provider: str, profile_model: str) -> None:
    os.environ["INTERGRAX_LLM_PROVIDER"] = profile_provider
    os.environ["INTERGRAX_LLM_MODEL"] = profile_model
    os.environ["INTERGRAX_DECISION_E2E_QUALIFICATION"] = "1"


class QualificationCohortExecutor:
    """Execute one cohort plan via the existing local qualification session."""

    def __init__(
        self,
        *,
        repo_root: Path,
        ollama_base_url: str = "http://127.0.0.1:11434",
        task_id: str = R6_TASK_ID,
        run_executor: QualificationRunExecutor | None = None,
        session_runner: _QualificationSessionRunner | None = None,
    ) -> None:
        self._repo_root = repo_root
        self._ollama_base_url = ollama_base_url
        self._task_id = task_id
        self._run_executor = run_executor or AiIncidentSingleRunExecutor()
        self._session_runner = session_runner or run_local_ai_incident_qualification

    async def execute_plan(
        self,
        plan: QualificationPlan,
        *,
        resume: bool = False,
        finalize_only: bool = False,
        repository_head_sha: str | None = None,
    ) -> CohortExecutionResult:
        profile_key = plan.profile.profile_key
        if plan.availability is ModelAvailability.MODEL_UNAVAILABLE:
            status = CohortExecutionStatus.MODEL_UNAVAILABLE
            return CohortExecutionResult(
                profile_key=profile_key,
                availability=ModelAvailability.MODEL_UNAVAILABLE,
                status=status,
                exit_code=QualificationCliExit.SUCCESS,
                session_state=None,
                failure=resolve_cohort_failure(
                    status=status.value,
                    exit_code=QualificationCliExit.SUCCESS,
                ),
            )
        if not plan.profile.digest:
            status = CohortExecutionStatus.BLOCKED_PRECONDITION
            return CohortExecutionResult(
                profile_key=profile_key,
                availability=plan.availability,
                status=status,
                exit_code=QualificationCliExit.BLOCKED_PRECONDITION,
                session_state=None,
                failure=resolve_cohort_failure(
                    status=status.value,
                    exit_code=QualificationCliExit.BLOCKED_PRECONDITION,
                ),
            )

        session_dir = plan.session_dir
        session_dir.mkdir(parents=True, exist_ok=True)
        resume_decision = decide_cohort_resume(
            session_dir,
            resume=resume,
            finalize_only=finalize_only,
        )
        if resume_decision.action is CohortResumeAction.SKIP_DUPLICATE_FINALIZED:
            status = CohortExecutionStatus.EXECUTED
            return CohortExecutionResult(
                profile_key=profile_key,
                availability=plan.availability,
                status=status,
                exit_code=QualificationCliExit.SUCCESS,
                session_state=resume_decision.session_state,
                failure=resolve_cohort_failure(
                    status=status.value,
                    exit_code=QualificationCliExit.SUCCESS,
                ),
            )
        if resume_decision.action is CohortResumeAction.BLOCK_RESUME_REQUIRED:
            status = CohortExecutionStatus.BLOCKED_PRECONDITION
            return CohortExecutionResult(
                profile_key=profile_key,
                availability=plan.availability,
                status=status,
                exit_code=QualificationCliExit.PARTIAL_OR_INVALID_SESSION,
                session_state=resume_decision.session_state,
                failure=resolve_cohort_failure(
                    status=status.value,
                    exit_code=QualificationCliExit.PARTIAL_OR_INVALID_SESSION,
                ),
            )

        _apply_profile_runtime_env(plan.profile.provider, plan.profile.model_name)
        bootstrap_qualification_environment(start_path=self._repo_root)
        head_sha = repository_head_sha or resolve_repository_head_sha(self._repo_root)
        spec, config_fp, frozen = build_qualification_spec_for_profile(
            self._repo_root,
            plan.profile,
            repository_head_sha=head_sha,
            run_count=plan.run_count,
        )
        probe = OllamaProviderIdentityProbe(
            OllamaProbeConfig(base_url=self._ollama_base_url)
        )
        result = await self._session_runner(
            repo_root=self._repo_root,
            session_dir=session_dir,
            spec=spec,
            config_fingerprint=config_fp,
            source_fingerprint=frozen.semantic_fingerprint(),
            provider_probe=probe,
            run_executor=self._run_executor,
            resume=resume,
            finalize_only=finalize_only,
            repository_head_sha=head_sha,
            task_id=self._task_id,
            temperature=plan.profile.temperature,
        )
        status = CohortExecutionStatus.EXECUTED
        return CohortExecutionResult(
            profile_key=profile_key,
            availability=plan.availability,
            status=status,
            exit_code=result.exit_code,
            session_state=result.session_state,
            failure=resolve_cohort_failure(
                status=status.value,
                exit_code=result.exit_code,
            ),
        )

    async def execute_plans(
        self,
        plans: tuple[QualificationPlan, ...],
        *,
        resume: bool = False,
        finalize_only: bool = False,
        repository_head_sha: str | None = None,
    ) -> tuple[CohortExecutionResult, ...]:
        head_sha = repository_head_sha or resolve_repository_head_sha(self._repo_root)
        results: list[CohortExecutionResult] = []
        for plan in plans:
            try:
                results.append(
                    await self.execute_plan(
                        plan,
                        resume=resume,
                        finalize_only=finalize_only,
                        repository_head_sha=head_sha,
                    )
                )
            except Exception as exc:
                status = CohortExecutionStatus.ISOLATED_FAILURE
                results.append(
                    CohortExecutionResult(
                        profile_key=plan.profile.profile_key,
                        availability=plan.availability,
                        status=status,
                        exit_code=QualificationCliExit.CRITICAL_SAFETY_FAILURE,
                        session_state=None,
                        failure=attribution_for_exception(exc),
                    )
                )
        return tuple(results)


def worst_cohort_exit_code(results: tuple[CohortExecutionResult, ...]) -> int:
    if not results:
        return int(QualificationCliExit.SUCCESS)
    return max(int(item.exit_code) for item in results)


__all__ = [
    "CohortExecutionResult",
    "CohortExecutionStatus",
    "QualificationCohortExecutor",
    "QualificationCohortResult",
    "QualificationPlan",
    "R6_LIVE_TASK_ID",
    "worst_cohort_exit_code",
]
