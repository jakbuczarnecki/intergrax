# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15B.1 semantic replay acceptance tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from intergrax.decision_system.qualification.taxonomy import DecisionFailureReason
from testing_support.decision_e2e.ds_e2e_15b1_semantic_replay import (
    GENUINE_SEMANTIC_RUN_INDICES,
    PROXY_ONLY_RUN_INDICES,
    SUCCESS_RUN_INDICES,
    UNSUPPORTED_COMPLETION_RUN_INDEX,
    replay_ai_incident_qualification_dataset,
    write_semantic_replay_artifacts,
)

pytestmark = pytest.mark.unit

SOURCE_DATASET = Path(".artifacts/qualification/DS-E2E-15A.1/runs.json")
OUTPUT_DIR = Path(".artifacts/qualification/DS-E2E-15B.1")


def _implementation_commit_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


@pytest.fixture(scope="module")
def semantic_replay_report():
    if not SOURCE_DATASET.is_file():
        pytest.skip("DS-E2E-15A.1 runs.json not available locally")
    report = replay_ai_incident_qualification_dataset(
        runs_json_path=SOURCE_DATASET,
        implementation_commit_sha=_implementation_commit_sha(),
    )
    write_semantic_replay_artifacts(report, output_dir=OUTPUT_DIR)
    return report


def test_proxy_only_runs_become_evaluator_pass(semantic_replay_report) -> None:
    by_index = {run["run_index"]: run for run in semantic_replay_report["runs"]}
    for run_index in PROXY_ONLY_RUN_INDICES:
        run = by_index[run_index]
        assert run["before_evaluator_passed"] is False
        assert run["after_evaluator_passed"] is True
        assert run["after_failures"] == ()


def test_genuine_semantic_runs_remain_failures(semantic_replay_report) -> None:
    by_index = {run["run_index"]: run for run in semantic_replay_report["runs"]}
    for run_index in GENUINE_SEMANTIC_RUN_INDICES:
        run = by_index[run_index]
        assert run["after_evaluator_passed"] is False
        assert "staffing_attendance_not_gathered" in run["after_failures"]


def test_success_and_unsupported_completion_unchanged(semantic_replay_report) -> None:
    by_index = {run["run_index"]: run for run in semantic_replay_report["runs"]}
    for run_index in SUCCESS_RUN_INDICES:
        run = by_index[run_index]
        assert run["before_evaluator_passed"] is True
        assert run["after_evaluator_passed"] is True
    unsupported = by_index[UNSUPPORTED_COMPLETION_RUN_INDEX]
    assert unsupported["before_evaluator_passed"] is False
    assert unsupported["after_evaluator_passed"] is False
    assert unsupported["before_reason"] == DecisionFailureReason.UNSUPPORTED_COMPLETION.value
    assert unsupported["after_reason"] == DecisionFailureReason.UNSUPPORTED_COMPLETION.value


def test_tool_use_deficiency_false_negatives_removed(semantic_replay_report) -> None:
    before = semantic_replay_report["before_distribution"].get(
        DecisionFailureReason.TOOL_USE_DEFICIENCY.value,
        0,
    )
    after = semantic_replay_report["after_distribution"].get(
        DecisionFailureReason.TOOL_USE_DEFICIENCY.value,
        0,
    )
    assert before == 13
    assert after == 0
    assert semantic_replay_report["after_distribution"].get(
        DecisionFailureReason.INSUFFICIENT_EVIDENCE_GATHERING.value,
        0,
    ) == len(GENUINE_SEMANTIC_RUN_INDICES)
