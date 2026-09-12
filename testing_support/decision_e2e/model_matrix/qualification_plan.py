# © Artur Czarnecki. All rights reserved.

"""Cohort layout and spec construction for multi-model qualification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from testing_support.decision_e2e.ai_incident_qualification_run import (
    CANONICAL_SCENARIO_INPUT_IDENTITY,
)
from testing_support.decision_e2e.local_ai_incident_qualification import (
    R4R1_QUANTIZATION,
    R4R1_SEMANTIC_VERIFICATION,
    build_generation_config_fingerprint,
    build_qualification_config_fingerprint,
    flattened_source_blob_paths,
    semantic_source_groups_for_r4r1,
)
from testing_support.decision_e2e.local_qualification_session.artifact_contract import (
    DEFAULT_REQUIRED_ARTIFACTS,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationExperimentIdentity,
    QualificationSpec,
    SourceFingerprintSnapshot,
    VersionMatchPolicy,
)
from testing_support.decision_e2e.local_qualification_session.source_fingerprint import (
    capture_semantic_source_fingerprint,
)
from testing_support.decision_e2e.local_qualification_session.ollama_probe import (
    OllamaProbeConfig,
)
from testing_support.decision_e2e.local_qualification_session.versioning import (
    parse_provider_runtime_version,
)
from testing_support.decision_e2e.model_matrix.availability import ModelAvailability
from testing_support.decision_e2e.model_matrix.model_execution_provider import (
    ModelExecutionProvider,
    OllamaModelExecutionProvider,
)
from testing_support.decision_e2e.model_matrix.profiles import ModelQualificationProfile
from testing_support.decision_e2e.model_matrix.source_freeze import TASK_ID
from testing_support.decision_e2e.scenario_qualification import AI_INCIDENT_SCENARIO_ID

DEFAULT_COHORT_RUN_COUNT = 20
R6_TASK_ID = TASK_ID


@dataclass(frozen=True, slots=True)
class ProfileCohortPlan:
    profile: ModelQualificationProfile
    session_dir: Path
    run_count: int
    availability: ModelAvailability


def qualification_artifact_root(repo_root: Path) -> Path:
    return repo_root / ".artifacts" / "qualification" / R6_TASK_ID


def profile_session_dir(repo_root: Path, profile: ModelQualificationProfile) -> Path:
    return qualification_artifact_root(repo_root) / profile.artifact_dir_name()


def summary_session_dir(repo_root: Path) -> Path:
    return qualification_artifact_root(repo_root) / "summary"


def cohort_checkpoint_indices(run_count: int) -> tuple[int, ...]:
    indices = tuple(index for index in (0, 10, 19) if 0 <= index < run_count)
    return indices or (0,)


def probe_profile_digest(
    profile: ModelQualificationProfile,
    *,
    execution_provider: ModelExecutionProvider | None = None,
    ollama_base_url: str = "http://127.0.0.1:11434",
    env_digest: str | None,
) -> tuple[str | None, ModelAvailability]:
    provider = execution_provider or OllamaModelExecutionProvider(
        OllamaProbeConfig(base_url=ollama_base_url)
    )
    return provider.resolve_profile_digest(profile, env_digest=env_digest)


def build_qualification_spec_for_profile(
    repo_root: Path,
    profile: ModelQualificationProfile,
    *,
    repository_head_sha: str,
    run_count: int = DEFAULT_COHORT_RUN_COUNT,
) -> tuple[QualificationSpec, str, SourceFingerprintSnapshot]:
    if not profile.digest:
        raise ValueError("profile.digest required before building qualification spec")
    groups = semantic_source_groups_for_r4r1()
    blob_paths = flattened_source_blob_paths(groups)
    frozen_source = capture_semantic_source_fingerprint(
        repo_root,
        semantic_source_groups=groups,
        repository_head_sha=repository_head_sha,
    )
    generation_fp = build_generation_config_fingerprint(
        temperature=profile.temperature,
        semantic_verification=R4R1_SEMANTIC_VERIFICATION,
        evaluator_max_iterations=profile.evaluator_iterations,
        max_decision_revisions=profile.revision_budget,
    )
    config_fp = build_qualification_config_fingerprint(
        provider=profile.provider,
        model=profile.model_name,
        generation_config_fingerprint=generation_fp,
        scenario_id=AI_INCIDENT_SCENARIO_ID,
        input_id=CANONICAL_SCENARIO_INPUT_IDENTITY,
    )
    runtime_version = parse_provider_runtime_version(profile.runtime_version)
    if runtime_version is None:
        raise ValueError(f"invalid profile.runtime_version: {profile.runtime_version}")
    identity = QualificationExperimentIdentity(
        provider_kind=profile.provider,
        provider_runtime_version=runtime_version,
        provider_runtime_version_policy=VersionMatchPolicy.EXACT,
        model_name=profile.model_name,
        model_digest=profile.digest,
        model_digest_policy=VersionMatchPolicy.EXACT,
        quantization=R4R1_QUANTIZATION,
        generation_config_fingerprint=generation_fp,
        scenario_id=AI_INCIDENT_SCENARIO_ID,
        input_id=CANONICAL_SCENARIO_INPUT_IDENTITY,
        source_fingerprint=frozen_source.semantic_fingerprint(),
        config_fingerprint=config_fp,
    )
    spec = QualificationSpec(
        experiment_identity=identity,
        run_count=run_count,
        required_artifacts=DEFAULT_REQUIRED_ARTIFACTS,
        source_blob_paths=blob_paths,
        semantic_source_groups=groups,
        max_evaluator_attempt_index=max(0, profile.evaluator_iterations - 1),
        source_checkpoint_run_indices=cohort_checkpoint_indices(run_count),
    )
    return spec, config_fp, frozen_source


def build_cohort_plans(
    repo_root: Path,
    profiles: tuple[ModelQualificationProfile, ...],
    *,
    run_count: int = DEFAULT_COHORT_RUN_COUNT,
    ollama_base_url: str,
    env_digest: str | None,
    execution_provider: ModelExecutionProvider | None = None,
) -> tuple[ProfileCohortPlan, ...]:
    from testing_support.decision_e2e.model_matrix.qualification_planner import (
        QualificationPlanner,
        default_planner,
    )

    planner = (
        QualificationPlanner(execution_provider)
        if execution_provider is not None
        else default_planner(ollama_base_url)
    )
    return planner.plan_cohorts(
        repo_root,
        profiles,
        run_count=run_count,
        env_digest=env_digest if env_digest else None,
    )


__all__ = [
    "DEFAULT_COHORT_RUN_COUNT",
    "ModelAvailability",
    "ProfileCohortPlan",
    "R6_TASK_ID",
    "build_cohort_plans",
    "build_qualification_spec_for_profile",
    "cohort_checkpoint_indices",
    "probe_profile_digest",
    "profile_session_dir",
    "qualification_artifact_root",
    "summary_session_dir",
]
