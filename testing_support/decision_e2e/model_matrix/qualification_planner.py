# © Artur Czarnecki. All rights reserved.

"""Qualification planner: registry profiles to cohort execution plans."""

from __future__ import annotations

from pathlib import Path

from testing_support.decision_e2e.model_matrix.model_execution_provider import (
    ModelExecutionProvider,
)
from testing_support.decision_e2e.model_matrix.profiles import ModelQualificationProfile
from testing_support.decision_e2e.model_matrix.qualification_plan import (
    DEFAULT_COHORT_RUN_COUNT,
    ProfileCohortPlan,
    profile_session_dir,
)


class QualificationPlanner:
    """Build validated cohort plans from the qualification registry."""

    def __init__(self, execution_provider: ModelExecutionProvider) -> None:
        self._execution = execution_provider

    def plan_cohorts(
        self,
        repo_root: Path,
        profiles: tuple[ModelQualificationProfile, ...],
        *,
        run_count: int = DEFAULT_COHORT_RUN_COUNT,
        env_digest: str | None = None,
    ) -> tuple[ProfileCohortPlan, ...]:
        plans: list[ProfileCohortPlan] = []
        for profile in profiles:
            digest, availability = self._execution.resolve_profile_digest(
                profile,
                env_digest=env_digest,
            )
            resolved = profile.with_digest(digest) if digest else profile
            plans.append(
                ProfileCohortPlan(
                    profile=resolved,
                    session_dir=profile_session_dir(repo_root, profile),
                    run_count=run_count,
                    availability=availability,
                )
            )
        return tuple(plans)


def default_planner(ollama_base_url: str) -> QualificationPlanner:
    from testing_support.decision_e2e.local_qualification_session.ollama_probe import (
        OllamaProbeConfig,
    )
    from testing_support.decision_e2e.model_matrix.model_execution_provider import (
        OllamaModelExecutionProvider,
    )

    return QualificationPlanner(
        OllamaModelExecutionProvider(OllamaProbeConfig(base_url=ollama_base_url))
    )


__all__ = ["QualificationPlanner", "default_planner"]
