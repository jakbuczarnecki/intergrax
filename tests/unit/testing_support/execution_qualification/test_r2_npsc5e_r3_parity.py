# © Artur Czarnecki. All rights reserved.

"""R2 parity and configuration proofs for NPSC-5E/R3 Final mandatory certification."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.execution_qualification.contracts import (
    QualificationRunConfig,
    QualificationRunStatus,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.coordinator import validate_and_run
from testing_support.execution_qualification.failure_report import format_execution_qualification_failure
from testing_support.execution_qualification.frozen_pytest_adapter import manifest_from_adapted_suites
from tests.unit.runtime.architecture import (
    test_npsc5e_final_recovery_plane_qualification_and_freeze as npsc5e_final,
)
from tests.unit.runtime.architecture import (
    test_npsc5e_r3_final_child_fanout_partial_recovery_qualification as r3_final,
)
from tests.unit.runtime.architecture.npsc5e_r3_final_execution_qualification import (
    NPSC5E_R3_CROSS_DB_EXCLUSIVE_RESOURCE_ID,
    NPSC5E_R3_EXECUTION_QUALIFICATION_DEFAULT_MAX_PARALLEL,
    NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID,
    build_npsc5e_r3_mandatory_projections,
    label_by_suite_id_from_projections,
    npsc5e_r3_qualification_run_config,
)

_CANCELLATION_EXPECTED_ARGS: tuple[str, ...] = (
    "tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py",
    "-k",
    "not survives_process_restart",
    "tests/unit/runtime/cancellation/test_p0c5a_explicit_terminal_wiring.py",
    "tests/unit/applications/test_task_control_governed_resume.py",
)

_R3_FINAL_TARGET = (
    "tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py"
)


def test_canonical_mandatory_source_matches_projection_count() -> None:
    source = r3_final._MANDATORY_SUITES
    adapted = build_npsc5e_r3_mandatory_projections(source)
    assert len(adapted) == len(source)
    assert len(adapted) == len(NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID)


def test_frozen_label_order_preserved() -> None:
    source = r3_final._MANDATORY_SUITES
    adapted = build_npsc5e_r3_mandatory_projections(source)
    assert tuple(entry.display_label for entry in adapted) == tuple(label for label, _ in source)


def test_pytest_argument_tuples_preserved_exactly() -> None:
    source = r3_final._MANDATORY_SUITES
    adapted = build_npsc5e_r3_mandatory_projections(source)
    for entry, (_, targets) in zip(adapted, source, strict=True):
        assert entry.suite.pytest_arguments == tuple(targets)


def test_cancellation_filter_parity() -> None:
    source = r3_final._MANDATORY_SUITES
    adapted = build_npsc5e_r3_mandatory_projections(source)
    cancellation = next(entry for entry in adapted if entry.display_label == "Cancellation")
    assert cancellation.suite.pytest_arguments == _CANCELLATION_EXPECTED_ARGS


def test_suite_ids_unique_and_deterministic() -> None:
    source = r3_final._MANDATORY_SUITES
    first = build_npsc5e_r3_mandatory_projections(source)
    second = build_npsc5e_r3_mandatory_projections(source)
    ids_first = tuple(entry.suite.suite_id for entry in first)
    ids_second = tuple(entry.suite.suite_id for entry in second)
    assert ids_first == ids_second
    assert len(set(ids_first)) == len(ids_first)


def test_r3_implementation_gate_exclusive_resource_only() -> None:
    source = r3_final._MANDATORY_SUITES
    adapted = build_npsc5e_r3_mandatory_projections(source)
    impl = next(entry for entry in adapted if entry.display_label == "R3 implementation gate")
    assert impl.suite.exclusive_resource_id == NPSC5E_R3_CROSS_DB_EXCLUSIVE_RESOURCE_ID
    for entry in adapted:
        if entry.display_label == "R3 implementation gate":
            continue
        assert entry.suite.exclusive_resource_id is None


def test_no_duplicate_or_missing_manifest_entries() -> None:
    source = r3_final._MANDATORY_SUITES
    adapted = build_npsc5e_r3_mandatory_projections(source)
    manifest = manifest_from_adapted_suites(adapted)
    assert len(manifest.suites) == len(source)
    assert {entry.display_label for entry in adapted} == {label for label, _ in source}


def test_npsc5e_final_invokes_r3_final_once() -> None:
    suites = npsc5e_final._MANDATORY_SUITES
    assert len(suites) == 1
    label, targets = suites[0]
    assert "R3" in label
    assert targets == [_R3_FINAL_TARGET]


def test_failure_projection_lists_all_non_pass_in_manifest_order(repo_root: Path) -> None:
    source = (
        (
            "Terminal",
            ["tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py"],
        ),
        (
            "Fan-out",
            ["tests/unit/agent_distribution/test_bounded_multi_agent_fanout.py::test_nonexistent_case"],
        ),
    )
    adapted = build_npsc5e_r3_mandatory_projections(source)
    manifest = manifest_from_adapted_suites(adapted)
    run_id = "r2-failure-projection"
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=repo_root / "build" / "qualification" / run_id,
        suite_timeout_seconds=300.0,
        run_id=run_id,
    )
    result = validate_and_run(manifest, config)
    assert result.status is QualificationRunStatus.FAIL
    non_pass = [r for r in result.suite_results if r.status is not QualificationSuiteStatus.PASS]
    assert len(non_pass) >= 1
    message = format_execution_qualification_failure(
        result,
        label_by_suite_id=label_by_suite_id_from_projections(adapted),
    )
    assert "[Terminal]" in message or "[Fan-out]" in message
    assert "log=" in message


def test_real_frozen_subset_parallel_qualification(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testing_support.execution_qualification.configuration import (
        EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV,
    )

    monkeypatch.delenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, raising=False)
    source = (
        (
            "Terminal",
            ["tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py"],
        ),
        (
            "Fan-out",
            ["tests/unit/agent_distribution/test_bounded_multi_agent_fanout.py"],
        ),
    )
    adapted = build_npsc5e_r3_mandatory_projections(source)
    manifest = manifest_from_adapted_suites(adapted)
    config = npsc5e_r3_qualification_run_config(repo_root, run_id="r2-real-subset")
    result = validate_and_run(manifest, config)
    assert result.status is QualificationRunStatus.PASS
    assert len(result.suite_results) == 2
    assert result.suite_results[0].suite_id == adapted[0].suite.suite_id
    assert result.suite_results[1].suite_id == adapted[1].suite.suite_id
    log_paths = {r.log_path for r in result.suite_results}
    assert len(log_paths) == 2
    for log_path in log_paths:
        assert log_path.is_file()
        assert str(config.run_artifact_root) in str(log_path)
    assert config.max_parallel == NPSC5E_R3_EXECUTION_QUALIFICATION_DEFAULT_MAX_PARALLEL
