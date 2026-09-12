# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from testing_support.decision_e2e.local_ai_incident_qualification import (
    QualificationCliExit,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationSessionState,
    SafetyGateOutcome,
)
from testing_support.decision_e2e.model_matrix.analysis import (
    FifteenKBEffectR6,
    classify_fifteen_kb_effect,
)
from testing_support.decision_e2e.model_matrix.availability import ModelAvailability
from testing_support.decision_e2e.model_matrix.matrix_artifact_contract import (
    MATRIX_ARTIFACT_SCHEMA_VERSION,
    QualificationArtifactProvider,
)
from testing_support.decision_e2e.model_matrix.profiles import ModelQualificationProfile
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionStatus,
    QualificationCohortExecutor,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_failure import (
    CohortFailureKind,
    failure_kind_for_exception,
)
from testing_support.decision_e2e.model_matrix.qualification_planner import (
    QualificationPlanner,
)
from testing_support.decision_e2e.model_matrix.qualification_plan import (
    ProfileCohortPlan,
    cohort_checkpoint_indices,
    profile_session_dir,
)
from testing_support.decision_e2e.model_matrix.registry import (
    QualificationRegistry,
    iter_qualification_profiles,
    qualification_matrix_version,
)
from testing_support.decision_e2e.model_matrix.source_freeze import (
    verify_model_matrix_source_freeze,
)


def test_registry_profiles_are_unique() -> None:
    keys = [profile.profile_key for profile in iter_qualification_profiles()]
    assert len(keys) == len(set(keys))
    assert "qwen2.5-14b" in keys
    assert "llama3.1-8b" in keys


def test_qualification_registry_version() -> None:
    assert QualificationRegistry.version() == qualification_matrix_version()


def test_cohort_checkpoint_indices_default_twenty() -> None:
    assert cohort_checkpoint_indices(20) == (0, 10, 19)


def test_source_freeze_passes_on_repository(repo_root: Path) -> None:
    report = verify_model_matrix_source_freeze(repo_root)
    gate_checks = [c for c in report.checks if c.name.startswith("gate:")]
    assert len(gate_checks) == 9
    assert all(check.passed for check in gate_checks)
    assert report.status.value == "PASS"


@pytest.mark.parametrize(
    ("overcommit", "revision", "typed", "repair", "expected"),
    [
        (0, 0, 0, 0, FifteenKBEffectR6.NOT_TRIGGERED),
        (2, 1, 1, 1, FifteenKBEffectR6.MODEL_PROVEN),
        (2, 0, 1, 0, FifteenKBEffectR6.FAIL),
        (2, 1, 0, 0, FifteenKBEffectR6.FAIL),
    ],
)
def test_classify_fifteen_kb_effect_cases(
    overcommit: int,
    revision: int,
    typed: int,
    repair: int,
    expected: FifteenKBEffectR6,
) -> None:
    effect = classify_fifteen_kb_effect(
        availability=ModelAvailability.AVAILABLE,
        model_overcommit_count=overcommit,
        revision_attempted=revision,
        typed_context_delivered=typed,
        repair_count=repair,
        total_runs=20,
        third_pass=0,
        reconciliation_leak=SafetyGateOutcome.PASS,
        alignment_events=20,
    )
    assert effect is expected


def test_profile_session_dir_under_r6_root(tmp_path: Path) -> None:
    profile = ModelQualificationProfile(
        profile_id="qwen2.5-14b",
        profile_key="qwen2.5-14b",
        provider="ollama",
        model_name="qwen2.5:14b",
        digest="sha256:abc",
        runtime_version="0.34.0",
        temperature=0.0,
        evaluator_iterations=2,
        revision_budget=0,
    )
    session_dir = profile_session_dir(tmp_path, profile)
    assert session_dir.name == "qwen2.5-14b"
    assert "DS-E2E-15J-L1.R6" in str(session_dir)


def test_matrix_artifact_contract_validation(tmp_path: Path) -> None:
    for name in QualificationArtifactProvider.required_artifact_names():
        if name == "checksum.json":
            continue
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    QualificationArtifactProvider.write_checksum_json(
        tmp_path,
        task_id="DS-E2E-15J-L1.R6",
        repository_head_sha="abc",
        source_fingerprint="fp",
        matrix_version="r6-v2",
    )
    QualificationArtifactProvider.write_manifest(
        tmp_path,
        tuple(
            name
            for name in QualificationArtifactProvider.required_artifact_names()
            if name != "artifact-manifest.txt"
        ),
    )
    result = QualificationArtifactProvider.validate_session(tmp_path)
    assert result.status.value == "COMPLETE"
    checksum = json.loads((tmp_path / "checksum.json").read_text(encoding="utf-8"))
    assert checksum["schema_version"] == MATRIX_ARTIFACT_SCHEMA_VERSION


def _sample_profile() -> ModelQualificationProfile:
    return ModelQualificationProfile(
        profile_id="qwen2.5-14b",
        profile_key="qwen2.5-14b",
        provider="ollama",
        model_name="qwen2.5:14b",
        digest="sha256:abc",
        runtime_version="0.34.0",
        temperature=0.0,
        evaluator_iterations=2,
        revision_budget=0,
    )


@pytest.mark.asyncio
async def test_cohort_executor_model_unavailable_skips_session(tmp_path: Path) -> None:
    profile = _sample_profile().with_digest("")
    plan = ProfileCohortPlan(
        profile=profile,
        session_dir=tmp_path / "qwen2.5-14b",
        run_count=20,
        availability=ModelAvailability.MODEL_UNAVAILABLE,
    )
    runner = AsyncMock()
    executor = QualificationCohortExecutor(
        repo_root=tmp_path,
        session_runner=runner,
    )
    result = await executor.execute_plan(plan)
    assert result.status is CohortExecutionStatus.MODEL_UNAVAILABLE
    assert result.exit_code is QualificationCliExit.SUCCESS
    runner.assert_not_called()


@pytest.mark.asyncio
async def test_cohort_executor_isolates_failures_across_plans(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    profile = _sample_profile()
    plans = (
        ProfileCohortPlan(
            profile=profile,
            session_dir=tmp_path / "ok",
            run_count=20,
            availability=ModelAvailability.AVAILABLE,
        ),
        ProfileCohortPlan(
            profile=profile.with_digest("sha256:other"),
            session_dir=tmp_path / "fail",
            run_count=20,
            availability=ModelAvailability.AVAILABLE,
        ),
    )
    invocations = 0

    async def _runner(**_kwargs):
        nonlocal invocations
        invocations += 1
        if invocations == 2:
            raise RuntimeError("simulated cohort failure")
        from testing_support.decision_e2e.local_ai_incident_qualification import (
            LocalQualificationOrchestrationResult,
        )

        return LocalQualificationOrchestrationResult(
            exit_code=QualificationCliExit.SUCCESS,
            session_state=QualificationSessionState.FINALIZED,
            executor_invocations=(0,),
        )

    executor = QualificationCohortExecutor(repo_root=repo_root, session_runner=_runner)
    results = await executor.execute_plans(plans)
    assert len(results) == 2
    assert results[0].status is CohortExecutionStatus.EXECUTED
    assert results[0].failure is None
    assert results[1].status is CohortExecutionStatus.ISOLATED_FAILURE
    assert results[1].exit_code is QualificationCliExit.CRITICAL_SAFETY_FAILURE
    assert results[1].failure is not None
    assert results[1].failure.kind is CohortFailureKind.UNKNOWN_FAILURE


@pytest.mark.asyncio
async def test_cohort_executor_forwards_resume_to_session_runner(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    plan = ProfileCohortPlan(
        profile=_sample_profile(),
        session_dir=tmp_path / "qwen2.5-14b",
        run_count=20,
        availability=ModelAvailability.AVAILABLE,
    )
    from testing_support.decision_e2e.local_ai_incident_qualification import (
        LocalQualificationOrchestrationResult,
    )

    runner = AsyncMock(
        return_value=LocalQualificationOrchestrationResult(
            exit_code=QualificationCliExit.SUCCESS,
            session_state=QualificationSessionState.FINALIZED,
            executor_invocations=(0,),
        )
    )
    executor = QualificationCohortExecutor(repo_root=repo_root, session_runner=runner)
    await executor.execute_plan(plan, resume=True, finalize_only=True)
    assert runner.call_args.kwargs["resume"] is True
    assert runner.call_args.kwargs["finalize_only"] is True


def test_planner_duplicate_profile_key_collides_session_dir(
    repo_root: Path,
) -> None:
    profile = _sample_profile()
    provider = MagicMock()
    provider.resolve_profile_digest.return_value = (
        "sha256:abc",
        ModelAvailability.AVAILABLE,
    )
    planner = QualificationPlanner(provider)
    plans = planner.plan_cohorts(
        repo_root,
        (profile, profile),
        env_digest="sha256:abc",
    )
    assert plans[0].session_dir == plans[1].session_dir


def test_planner_distinct_profile_keys_isolate_artifact_dirs(
    repo_root: Path,
) -> None:
    provider = MagicMock()
    provider.resolve_profile_digest.return_value = (
        "sha256:abc",
        ModelAvailability.AVAILABLE,
    )
    planner = QualificationPlanner(provider)
    a = _sample_profile()
    b = a.with_digest("sha256:abc").__class__(
        profile_id="llama3.1-8b",
        profile_key="llama3.1-8b",
        provider=a.provider,
        model_name="llama3.1:8b",
        digest="sha256:abc",
        runtime_version=a.runtime_version,
        temperature=a.temperature,
        evaluator_iterations=a.evaluator_iterations,
        revision_budget=a.revision_budget,
    )
    plans = planner.plan_cohorts(repo_root, (a, b), env_digest="sha256:abc")
    assert plans[0].session_dir != plans[1].session_dir


def test_provider_failure_not_used_for_unknown_exceptions() -> None:
    assert (
        failure_kind_for_exception(RuntimeError("simulated"))
        is CohortFailureKind.UNKNOWN_FAILURE
    )
    assert (
        failure_kind_for_exception(ConnectionError("down"))
        is CohortFailureKind.PROVIDER_FAILURE
    )


def test_cohort_executor_boundary_excludes_matrix_analysis() -> None:
    import testing_support.decision_e2e.model_matrix.qualification_cohort_executor as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "run_multi_model_qualification_analysis" not in source
    assert "analysis.json" not in source


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]
