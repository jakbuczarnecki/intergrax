# © Artur Czarnecki. All rights reserved.

"""NPSC-5F Final — single-invocation mandatory regression matrix (qualification only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from testing_support.npsc5f_r4_regression_matrix import (
    MANDATORY_REGRESSION_SUITES as _R4_MANDATORY_REGRESSION_SUITES,
    flatten_regression_targets,
)

# Exclude orchestrators and nested ``uv run pytest`` / ruff / pyright fan-out tests.
_PYTEST_MATRIX_KEXPR = (
    "not test_mandatory_frozen_suite_passes and "
    "not test_mandatory_frozen_suites_pass_via_parallel_qualification and "
    "not test_mandatory_regression_matrix_passes and "
    "not test_r4_mandatory_regression_matrix and "
    "not test_npsc5f_final_mandatory_regression_matrix_passes and "
    "not test_pre_existing_cancellation_fixture_same_root_cause and "
    "not test_pre_existing_partial_results_fixture_unchanged_baseline and "
    "not test_ruff_recovery_surfaces_and_final_test and "
    "not test_pyright_recovery_surfaces_and_final_test and "
    "not test_terminal_cancellation_survives_process_restart"
)

_NPSC5F_FINAL_EXTRA_SUITES: tuple[tuple[str, list[str]], ...] = (
    (
        "Recovery",
        [
            "tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py",
            "tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py",
            "tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py",
        ],
    ),
    (
        "NPSC-5E Final",
        ["tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"],
    ),
    (
        "HITL R3",
        ["tests/unit/runtime/architecture/test_npsc5d_r3_governed_continuation.py"],
    ),
    (
        "Child execution",
        [
            "tests/unit/runtime/execution/test_child_execution.py",
            "tests/unit/runtime/execution/authority/test_child_execution_authority_policy.py",
        ],
    ),
    (
        "Checkpoint",
        [
            "tests/unit/runtime/long_running/test_checkpoint_store.py",
            "tests/unit/runtime/long_running/test_runtime_checkpoint.py",
        ],
    ),
    (
        "Retry",
        ["tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py"],
    ),
    (
        "Cancellation",
        [
            "tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py",
            "tests/unit/runtime/cancellation/test_p0c5a_explicit_terminal_wiring.py",
        ],
    ),
    (
        "Evidence",
        [
            "tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py",
            "tests/unit/runtime/architecture/test_npsc5f_enterprise_evidence_certification.py",
            "tests/unit/runtime/architecture/test_npsc5f_r1_durable_evidence_persistence_boundary_resignoff.py",
            "tests/unit/runtime/events/test_evidence_persistence_boundary.py",
        ],
    ),
    (
        "NPSC-5D Final",
        ["tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"],
    ),
    (
        "NPSC-5F Final drift sentinel",
        ["tests/unit/testing_support/test_npsc5f_final_protected_drift.py"],
    ),
)

MANDATORY_REGRESSION_SUITES: tuple[tuple[str, list[str]], ...] = (
    *_R4_MANDATORY_REGRESSION_SUITES,
    *_NPSC5F_FINAL_EXTRA_SUITES,
)


def run_mandatory_regression_matrix(
    repo_root: Path,
    *,
    suites: tuple[tuple[str, list[str]], ...] = MANDATORY_REGRESSION_SUITES,
) -> subprocess.CompletedProcess[str]:
    """Run the full matrix in one ``uv run pytest`` invocation (no per-suite subprocess fan-out)."""
    targets = flatten_regression_targets(suites)
    return subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            *targets,
            "-q",
            "--tb=line",
            "-k",
            _PYTEST_MATRIX_KEXPR,
            "-p",
            "no:xdist",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
