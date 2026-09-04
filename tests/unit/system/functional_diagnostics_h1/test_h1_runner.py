# © Artur Czarnecki. All rights reserved.

"""Unit tests for DIAG-FUNCTIONAL-H1 runner semantics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
)
from tests.system.functional_diagnostics_h1.reporting import aggregate_overall_verdict
from tests.system.functional_diagnostics_h1.subprocess_pytest import classify_pytest_exit

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FAIL_FIXTURE = _REPO_ROOT / "tests" / "unit" / "system" / "functional_diagnostics_h1" / "fixtures" / "synthetic_fail_test.py"
_BLOCKED_FIXTURE = _REPO_ROOT / "tests" / "unit" / "system" / "functional_diagnostics_h1" / "fixtures" / "synthetic_blocked_preflight.py"


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
    gates = (
        GateResult(HealthGateId.H1_A_COLLECTION, HealthVerdict.PASS, "ok"),
        GateResult(HealthGateId.H1_B_CORE_HEALTH, HealthVerdict.PASS, "ok"),
    )
    core, real, overall = aggregate_overall_verdict(gates, real_service_blocked=True)
    assert core is HealthVerdict.PASS
    assert real is HealthVerdict.BLOCKED
    assert overall is HealthVerdict.PASS


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
