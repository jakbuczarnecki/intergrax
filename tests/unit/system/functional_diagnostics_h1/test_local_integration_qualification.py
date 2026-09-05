# © Artur Czarnecki. All rights reserved.

"""Unit tests for DIAG-H1-K local integration qualification infrastructure."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.system.functional_diagnostics_h1.inventory import LOCAL_INTEGRATION_TARGETS
from tests.system.functional_diagnostics_h1.local_integration import (
    evaluate_local_integration_repeatability,
    evaluate_local_integration_suite,
    gate_result_from_local_integration_run,
    run_h1_k_local_integration,
)
from tests.system.functional_diagnostics_h1.local_integration_reporting import (
    build_local_integration_human_report,
    validate_local_integration_report_integrity,
)
from tests.system.functional_diagnostics_h1.models import (
    HealthGateId,
    HealthVerdict,
    H1_K_QUALIFICATION_ID,
    LocalIntegrationDependencyClass,
    LocalIntegrationQualificationReport,
    LocalIntegrationRunResult,
    LocalIntegrationSuiteResult,
    PytestSubprocessResult,
    QualificationRepositoryState,
)
from tests.system.functional_diagnostics_h1.qualification_spec import H1_K_QUALIFICATION_SPEC

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SYNTHETIC_TARGET_A = "tests/unit/system/functional_diagnostics_h1/fixtures/synthetic_pass_a.py"
_SYNTHETIC_TARGET_B = "tests/unit/system/functional_diagnostics_h1/fixtures/synthetic_pass_b.py"


def _passing_pytest_result(collected: int = 2, passed: int = 2) -> PytestSubprocessResult:
    return PytestSubprocessResult(
        exit_code=0,
        collected_count=collected,
        passed=passed,
        failed=0,
        skipped=0,
        xfailed=0,
        xpassed=0,
        errors=0,
        collection_errors=0,
        stdout_tail="",
        stderr_tail="",
        duration_seconds=1.0,
    )


def _failing_pytest_result() -> PytestSubprocessResult:
    return PytestSubprocessResult(
        exit_code=1,
        collected_count=1,
        passed=0,
        failed=1,
        skipped=0,
        xfailed=0,
        xpassed=0,
        errors=0,
        collection_errors=0,
        stdout_tail="",
        stderr_tail="",
        duration_seconds=1.0,
    )


def _suite(
    target: str,
    *,
    collected: int = 2,
    passed: int = 2,
    failed: int = 0,
    errors: int = 0,
    skipped: int = 0,
    xfailed: int = 0,
    xpassed: int = 0,
    verdict: HealthVerdict = HealthVerdict.PASS,
) -> LocalIntegrationSuiteResult:
    return LocalIntegrationSuiteResult(
        target=target,
        collected=collected,
        passed=passed,
        failed=failed,
        skipped=skipped,
        xfailed=xfailed,
        xpassed=xpassed,
        errors=errors,
        collection_errors=0,
        exit_code=0 if verdict is HealthVerdict.PASS else 1,
        duration_seconds=1.0,
        verdict=verdict,
        dependency_class=LocalIntegrationDependencyClass.LOCAL_DETERMINISTIC,
    )


def _run(
    run_index: int,
    suites: tuple[LocalIntegrationSuiteResult, ...],
) -> LocalIntegrationRunResult:
    verdict = HealthVerdict.PASS
    if any(item.verdict is not HealthVerdict.PASS for item in suites):
        verdict = HealthVerdict.FAILED
    collected_values = [item.collected for item in suites if item.collected is not None]
    total_collected = sum(collected_values) if collected_values else None
    return LocalIntegrationRunResult(
        run_index=run_index,
        suite_results=suites,
        verdict=verdict,
        total_collected=total_collected,
        total_passed=sum(item.passed for item in suites),
        total_failed=sum(item.failed for item in suites),
        total_errors=sum(item.errors for item in suites),
    )


def test_one_failed_suite_produces_run_failed() -> None:
    suite = evaluate_local_integration_suite("target.py", _failing_pytest_result())
    assert suite.verdict is HealthVerdict.FAILED


def test_passing_suite_produces_run_pass() -> None:
    suite = evaluate_local_integration_suite("target.py", _passing_pytest_result())
    assert suite.verdict is HealthVerdict.PASS


def test_unexpected_skip_fails_suite() -> None:
    result = PytestSubprocessResult(
        exit_code=0,
        collected_count=2,
        passed=1,
        failed=0,
        skipped=1,
        xfailed=0,
        xpassed=0,
        errors=0,
        collection_errors=0,
        stdout_tail="",
        stderr_tail="",
        duration_seconds=1.0,
    )
    suite = evaluate_local_integration_suite("target.py", result)
    assert suite.verdict is HealthVerdict.FAILED


def test_one_failed_run_produces_qualification_failed() -> None:
    runs = (
        _run(1, (_suite("a.py"),)),
        _run(2, (_suite("a.py", verdict=HealthVerdict.FAILED, failed=1),)),
        _run(3, (_suite("a.py"),)),
    )
    repeatability, findings = evaluate_local_integration_repeatability(runs)
    assert repeatability is HealthVerdict.FAILED
    assert findings


def test_count_variance_fails_repeatability() -> None:
    runs = (
        _run(1, (_suite("a.py", collected=2, passed=2),)),
        _run(2, (_suite("a.py", collected=3, passed=3),)),
        _run(3, (_suite("a.py", collected=2, passed=2),)),
    )
    repeatability, findings = evaluate_local_integration_repeatability(runs)
    assert repeatability is HealthVerdict.FAILED
    assert any("collected_variance" in item for item in findings)


def test_target_set_variance_fails_repeatability() -> None:
    runs = (
        _run(1, (_suite("a.py"), _suite("b.py"))),
        _run(2, (_suite("a.py"),)),
        _run(3, (_suite("a.py"), _suite("b.py"))),
    )
    repeatability, findings = evaluate_local_integration_repeatability(runs)
    assert repeatability is HealthVerdict.FAILED
    assert any("target_set_variance" in item for item in findings)


def test_stable_runs_pass_repeatability() -> None:
    suites = (_suite("a.py"), _suite("b.py"))
    runs = (_run(1, suites), _run(2, suites), _run(3, suites))
    repeatability, findings = evaluate_local_integration_repeatability(runs)
    assert repeatability is HealthVerdict.PASS
    assert findings == ()


def test_dirty_start_failed_precondition() -> None:
    from tests.system.functional_diagnostics_h1.local_integration_qualification import (
        run_h1_k_qualification,
    )

    artifact_dir = _REPO_ROOT / ".tmp" / "session" / "diag-h1-k-qualification-r1-unit"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with patch(
        "tests.system.functional_diagnostics_h1.local_integration_qualification.capture_qualification_repository_state",
        return_value=QualificationRepositoryState(
            head_sha="abc",
            origin_development_sha="abc",
            working_tree_clean=False,
        ),
    ):
        exit_code = run_h1_k_qualification(artifact_dir=artifact_dir)
    assert exit_code == 3


def test_head_not_equal_origin_failed_precondition() -> None:
    from tests.system.functional_diagnostics_h1.repository_state import (
        assert_qualification_repository_state,
    )

    state = QualificationRepositoryState(
        head_sha="abc",
        origin_development_sha="def",
        working_tree_clean=True,
    )
    verdict, violations = assert_qualification_repository_state(state)
    assert verdict is HealthVerdict.FAILED_PRECONDITION
    assert violations


def test_blocking_finding_and_pass_impossible() -> None:
    report = LocalIntegrationQualificationReport(
        qualification_id=H1_K_QUALIFICATION_ID,
        schema_version="diag_h1_k_local_integration_v1",
        tested_sha="abc",
        start_head="abc",
        final_head="abc",
        origin_development_at_start="abc",
        origin_development_at_end="abc",
        working_tree_clean_at_start=True,
        working_tree_clean_at_end=True,
        repository_precondition=HealthVerdict.PASS,
        repository_postcondition=HealthVerdict.PASS,
        runs=(_run(1, (_suite("a.py"),)),),
        repeatability_verdict=HealthVerdict.PASS,
        overall_verdict=HealthVerdict.PASS,
        blocking_findings=("stale finding",),
        timestamp="2026-01-01T00:00:00+00:00",
    )
    integrity, violations = validate_local_integration_report_integrity(report)
    assert integrity is HealthVerdict.FAILED
    assert "blocking_findings_coexist_with_overall_pass" in violations


def test_gate_projection_uses_execution_core() -> None:
    run = _run(1, (_suite(LOCAL_INTEGRATION_TARGETS[0]),))
    gate = gate_result_from_local_integration_run(run)
    assert gate.gate_id is HealthGateId.H1_K_LOCAL_INTEGRATION
    assert gate.verdict is HealthVerdict.PASS
    assert LOCAL_INTEGRATION_TARGETS[0] in gate.details[0]


def test_synthetic_target_injection_without_registry_change() -> None:
    synthetic_targets = (_SYNTHETIC_TARGET_A, _SYNTHETIC_TARGET_B)
    with patch(
        "tests.system.functional_diagnostics_h1.local_integration.run_pytest_subprocess",
        return_value=_passing_pytest_result(),
    ):
        run = run_h1_k_local_integration(run_index=1, targets=synthetic_targets)
    assert len(run.suite_results) == 2
    assert run.verdict is HealthVerdict.PASS


def test_h1_k_spec_artifact_directory() -> None:
    assert (
        H1_K_QUALIFICATION_SPEC.artifact_directory.as_posix()
        == ".tmp/session/diag-h1-k-qualification-r1"
    )


def test_human_report_is_projection_of_machine_report() -> None:
    suites = tuple(_suite(target) for target in LOCAL_INTEGRATION_TARGETS)
    report = LocalIntegrationQualificationReport(
        qualification_id=H1_K_QUALIFICATION_ID,
        schema_version="diag_h1_k_local_integration_v1",
        tested_sha="abc123",
        start_head="abc123",
        final_head="abc123",
        origin_development_at_start="abc123",
        origin_development_at_end="abc123",
        working_tree_clean_at_start=True,
        working_tree_clean_at_end=True,
        repository_precondition=HealthVerdict.PASS,
        repository_postcondition=HealthVerdict.PASS,
        runs=(
            _run(1, suites),
            _run(2, suites),
            _run(3, suites),
        ),
        repeatability_verdict=HealthVerdict.PASS,
        overall_verdict=HealthVerdict.PASS,
        blocking_findings=(),
        timestamp="2026-01-01T00:00:00+00:00",
    )
    markdown = build_local_integration_human_report(
        report,
        artifact_json_path=".tmp/session/diag-h1-k-qualification-r1/qualification-report.json",
    )
    assert "Run 1 PASS" in markdown
    assert "Repeatability PASS" in markdown
    assert "Overall PASS" in markdown
    for target in LOCAL_INTEGRATION_TARGETS:
        assert target in markdown
