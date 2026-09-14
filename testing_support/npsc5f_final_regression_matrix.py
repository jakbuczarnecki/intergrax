# © Artur Czarnecki. All rights reserved.

"""NPSC-5F Final — single-invocation mandatory regression matrix (qualification only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES,
)
from testing_support.npsc5f_r4_regression_matrix import flatten_regression_targets

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
    "not test_pyright_recovery_surfaces_and_final_test"
)

MANDATORY_REGRESSION_SUITES: tuple[tuple[str, list[str]], ...] = (
    NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES
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
