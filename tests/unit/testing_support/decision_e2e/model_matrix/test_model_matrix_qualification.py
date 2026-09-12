# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from testing_support.decision_e2e.local_qualification_session.contracts import SafetyGateOutcome
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
from testing_support.decision_e2e.model_matrix.qualification_plan import (
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


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]
