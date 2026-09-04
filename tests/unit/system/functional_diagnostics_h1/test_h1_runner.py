# © Artur Czarnecki. All rights reserved.

"""Unit tests for DIAG-FUNCTIONAL-H1 runner semantics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.system.functional_diagnostics_h1.composition import build_h1_qualification_families
from tests.system.functional_diagnostics_h1.inventory import (
    build_diagnostic_test_inventory,
    build_invariant_ownership_matrix,
    verify_invariant_owners,
)
from tests.system.functional_diagnostics_h1.models import (
    ExternalDependencyState,
    GateResult,
    HealthGateId,
    HealthVerdict,
    PytestSubprocessResult,
    QualificationRepositoryState,
)
from tests.system.functional_diagnostics_h1.reporting import (
    aggregate_overall_verdict,
    calculate_health_verdict,
    gate_h1_j_report_integrity,
)
from tests.system.functional_diagnostics_h1.repository_state import assert_qualification_repository_state
from tests.system.functional_diagnostics_h1.subprocess_pytest import classify_pytest_exit

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FAIL_FIXTURE = _REPO_ROOT / "tests" / "unit" / "system" / "functional_diagnostics_h1" / "fixtures" / "synthetic_fail_test.py"
_BLOCKED_FIXTURE = _REPO_ROOT / "tests" / "unit" / "system" / "functional_diagnostics_h1" / "fixtures" / "synthetic_blocked_preflight.py"


def _pass_gates(*exclude: HealthGateId) -> tuple[GateResult, ...]:
    excluded = frozenset(exclude)
    all_ids = (
        HealthGateId.H1_A_COLLECTION,
        HealthGateId.H1_B_CORE_HEALTH,
        HealthGateId.H1_C_REPEATABILITY,
        HealthGateId.H1_D_INVARIANT_COVERAGE,
        HealthGateId.H1_E_SKIP_XFAIL_HONESTY,
        HealthGateId.H1_F_EXTERNAL_DEPENDENCY,
        HealthGateId.H1_G_RUNNER_INTEGRITY,
        HealthGateId.H1_H_STALE_DEAD,
        HealthGateId.H1_I_SUPERSESSION,
        HealthGateId.H1_K_LOCAL_INTEGRATION,
    )
    return tuple(
        GateResult(gate_id, HealthVerdict.PASS, "ok")
        for gate_id in all_ids
        if gate_id not in excluded
    )


def test_inventory_non_empty() -> None:
    inventory = build_diagnostic_test_inventory()
    assert len(inventory) > 0


def test_invariant_owners_exist() -> None:
    missing = verify_invariant_owners(_REPO_ROOT)
    assert missing == ()


def test_h1_registry_extensibility_without_core_modification() -> None:
    families = build_h1_qualification_families()
    assert any(item.family.value == "H1" for item in families)


def test_aggregate_overall_verdict_core_pass_real_blocked() -> None:
    gates = _pass_gates()
    core, real, overall = aggregate_overall_verdict(gates, real_service_blocked=True)
    assert core is HealthVerdict.PASS
    assert real is HealthVerdict.BLOCKED
    assert overall is HealthVerdict.PASS


def test_local_failed_mandatory_gate_produces_overall_failed() -> None:
    gates = _pass_gates(HealthGateId.H1_K_LOCAL_INTEGRATION) + (
        GateResult(
            HealthGateId.H1_K_LOCAL_INTEGRATION,
            HealthVerdict.FAILED,
            "local integration failed",
        ),
    )
    core, _real, overall = calculate_health_verdict(gates, real_service_blocked=True)
    assert core is HealthVerdict.FAILED
    assert overall is HealthVerdict.FAILED


def test_core_failed_mandatory_gate_produces_overall_failed() -> None:
    gates = _pass_gates(HealthGateId.H1_B_CORE_HEALTH) + (
        GateResult(HealthGateId.H1_B_CORE_HEALTH, HealthVerdict.FAILED, "core failed"),
    )
    _core, _real, overall = calculate_health_verdict(gates, real_service_blocked=False)
    assert overall is HealthVerdict.FAILED


def test_report_integrity_rejects_failed_gate_with_pass_overall() -> None:
    gates = _pass_gates(HealthGateId.H1_K_LOCAL_INTEGRATION) + (
        GateResult(HealthGateId.H1_K_LOCAL_INTEGRATION, HealthVerdict.FAILED, "local failed"),
    )
    integrity = gate_h1_j_report_integrity(
        gate_results=gates,
        calculated_overall=HealthVerdict.PASS,
        blocking_findings=("local failed",),
        start_head="abc",
        final_head="abc",
    )
    assert integrity.verdict is HealthVerdict.FAILED


def test_report_integrity_rejects_blocking_findings_with_pass_overall() -> None:
    gates = _pass_gates()
    integrity = gate_h1_j_report_integrity(
        gate_results=gates,
        calculated_overall=HealthVerdict.PASS,
        blocking_findings=("stale finding",),
        start_head="abc",
        final_head="abc",
    )
    assert integrity.verdict is HealthVerdict.FAILED


def test_external_blocked_only_preserves_core_pass() -> None:
    gates = _pass_gates()
    core, real, overall = calculate_health_verdict(gates, real_service_blocked=True)
    assert core is HealthVerdict.PASS
    assert real is HealthVerdict.BLOCKED
    assert overall is HealthVerdict.PASS


def test_dirty_working_tree_refuses_qualification() -> None:
    state = QualificationRepositoryState(
        head_sha="abc123",
        origin_development_sha="abc123",
        working_tree_clean=False,
    )
    verdict, violations = assert_qualification_repository_state(state)
    assert verdict is HealthVerdict.FAILED_PRECONDITION
    assert "working_tree_not_clean" in violations


def test_head_not_equal_origin_refuses_qualification() -> None:
    state = QualificationRepositoryState(
        head_sha="abc123",
        origin_development_sha="def456",
        working_tree_clean=True,
    )
    verdict, violations = assert_qualification_repository_state(state)
    assert verdict is HealthVerdict.FAILED_PRECONDITION
    assert any("head_not_pushed" in item for item in violations)


def test_head_change_during_qualification_fails_integrity() -> None:
    gates = _pass_gates()
    integrity = gate_h1_j_report_integrity(
        gate_results=gates,
        calculated_overall=HealthVerdict.PASS,
        blocking_findings=(),
        start_head="abc",
        final_head="def",
    )
    assert integrity.verdict is HealthVerdict.FAILED


def test_classify_pytest_exit_failed_not_blocked() -> None:
    result = PytestSubprocessResult(
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
        duration_seconds=0.1,
    )
    assert classify_pytest_exit(result) == "FAILED"


def test_synthetic_failing_subprocess_classified_failed() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(_FAIL_FIXTURE),
            "-q",
            "--tb=no",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "FAILED" in completed.stdout or completed.returncode == 1


def test_synthetic_blocked_preflight_not_pass() -> None:
    completed = subprocess.run(
        [sys.executable, str(_BLOCKED_FIXTURE)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "BLOCKED" in completed.stdout


def test_invariant_matrix_has_normative_owners() -> None:
    matrix = build_invariant_ownership_matrix()
    assert len(matrix) >= 20
    for owner in matrix:
        assert owner.normative_owner
        assert (_REPO_ROOT / owner.normative_owner).exists()


def test_external_dependency_state_enum_values() -> None:
    assert ExternalDependencyState.BLOCKED_MISSING_CREDENTIAL.value.startswith("BLOCKED")


def test_h1_r2_runner_refuses_dirty_tree() -> None:
    from tests.system.functional_diagnostics_h1.models import H1_R2_QUALIFICATION_ID
    from tests.system.functional_diagnostics_h1.runner import run_h1_qualification

    artifact_dir = _REPO_ROOT / ".tmp" / "session" / "diag-functional-h1-r2" / "unit-test-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    human_doc = artifact_dir / "report.md"
    with patch(
        "tests.system.functional_diagnostics_h1.runner.capture_qualification_repository_state",
        return_value=QualificationRepositoryState(
            head_sha="abc",
            origin_development_sha="abc",
            working_tree_clean=False,
        ),
    ):
        exit_code = run_h1_qualification(
            qualification_id=H1_R2_QUALIFICATION_ID,
            artifact_dir=artifact_dir,
            human_doc_path=human_doc,
        )
    assert exit_code == 3
