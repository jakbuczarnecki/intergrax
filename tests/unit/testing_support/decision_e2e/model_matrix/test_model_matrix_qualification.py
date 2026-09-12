# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.decision_e2e.local_qualification_session.contracts import SafetyGateOutcome
from testing_support.decision_e2e.model_matrix.analysis import (
    FifteenKBEffectR6,
    classify_fifteen_kb_effect,
)
from testing_support.decision_e2e.model_matrix.profiles import ModelQualificationProfile
from testing_support.decision_e2e.model_matrix.qualification_plan import (
    cohort_checkpoint_indices,
    profile_session_dir,
)
from testing_support.decision_e2e.model_matrix.registry import iter_qualification_profiles
from testing_support.decision_e2e.model_matrix.source_freeze import (
    verify_model_matrix_source_freeze,
)


def test_registry_profiles_are_unique() -> None:
    keys = [profile.profile_key for profile in iter_qualification_profiles()]
    assert len(keys) == len(set(keys))
    assert "qwen2.5-14b" in keys


def test_cohort_checkpoint_indices_default_twenty() -> None:
    assert cohort_checkpoint_indices(20) == (0, 10, 19)


def test_source_freeze_passes_on_repository(repo_root: Path) -> None:
    report = verify_model_matrix_source_freeze(repo_root)
    group_checks = [c for c in report.checks if c.name.startswith("group:")]
    assert group_checks
    assert all(check.detail == "IDENTICAL" for check in group_checks)
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
    from testing_support.decision_e2e.model_matrix.qualification_plan import ModelAvailability

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
        profile_key="qwen2.5-14b",
        provider="ollama",
        model_id="qwen2.5:14b",
        digest="sha256:abc",
        temperature=0.0,
    )
    session_dir = profile_session_dir(tmp_path, profile)
    assert session_dir.name == "qwen2.5-14b"
    assert "DS-E2E-15J-L1.R6" in str(session_dir)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]
