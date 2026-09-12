# © Artur Czarnecki. All rights reserved.

"""Cohort layout and spec construction for multi-model qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from testing_support.decision_e2e.ai_incident_qualification_run import (
    CANONICAL_SCENARIO_INPUT_IDENTITY,
)
from testing_support.decision_e2e.local_ai_incident_qualification import (
    R4R1_EVALUATOR_MAX_ITERATIONS,
    R4R1_MAX_DECISION_REVISIONS,
    R4R1_QUANTIZATION,
    R4R1_RUNTIME_VERSION,
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
from testing_support.decision_e2e.local_qualification_session.ollama_probe import (
    OllamaProbeConfig,
    probe_ollama_runtime_identity,
)
from testing_support.decision_e2e.local_qualification_session.source_fingerprint import (
    capture_semantic_source_fingerprint,
)
from testing_support.decision_e2e.model_matrix.profiles import ModelQualificationProfile
from testing_support.decision_e2e.model_matrix.source_freeze import TASK_ID
from testing_support.decision_e2e.scenario_qualification import AI_INCIDENT_SCENARIO_ID

DEFAULT_COHORT_RUN_COUNT = 20
R6_TASK_ID = TASK_ID


class ModelAvailability(StrEnum):
    AVAILABLE = "AVAILABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


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
    ollama_base_url: str,
    env_digest: str | None,
) -> tuple[str | None, ModelAvailability]:
    if env_digest:
        return env_digest, ModelAvailability.AVAILABLE
    observed = probe_ollama_runtime_identity(
        OllamaProbeConfig(base_url=ollama_base_url),
        model_name=profile.model_id,
    )
    if observed is None or observed.model_digest is None:
        return None, ModelAvailability.MODEL_UNAVAILABLE
    return observed.model_digest, ModelAvailability.AVAILABLE


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
        evaluator_max_iterations=R4R1_EVALUATOR_MAX_ITERATIONS,
        max_decision_revisions=R4R1_MAX_DECISION_REVISIONS,
    )
    config_fp = build_qualification_config_fingerprint(
        provider=profile.provider,
        model=profile.model_id,
        generation_config_fingerprint=generation_fp,
        scenario_id=AI_INCIDENT_SCENARIO_ID,
        input_id=CANONICAL_SCENARIO_INPUT_IDENTITY,
    )
    identity = QualificationExperimentIdentity(
        provider_kind=profile.provider,
        provider_runtime_version=R4R1_RUNTIME_VERSION,
        provider_runtime_version_policy=VersionMatchPolicy.EXACT,
        model_name=profile.model_id,
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
        max_evaluator_attempt_index=max(0, R4R1_EVALUATOR_MAX_ITERATIONS - 1),
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
) -> tuple[ProfileCohortPlan, ...]:
    plans: list[ProfileCohortPlan] = []
    for profile in profiles:
        digest, availability = probe_profile_digest(
            profile,
            ollama_base_url=ollama_base_url,
            env_digest=env_digest if env_digest else None,
        )
        resolved = profile
        if digest:
            resolved = profile.with_digest(digest)
        plans.append(
            ProfileCohortPlan(
                profile=resolved,
                session_dir=profile_session_dir(repo_root, profile),
                run_count=run_count,
                availability=availability,
            )
        )
    return tuple(plans)


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
