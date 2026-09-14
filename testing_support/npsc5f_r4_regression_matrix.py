# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R4 Final — single-invocation mandatory regression matrix targets (qualification only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5F_R4_MANDATORY_REGRESSION_SUITES,
)

# Exclude orchestrator tests that spawn nested ``uv run pytest`` (process explosion / recursion).
_PYTEST_MATRIX_KEXPR = (
    "not test_mandatory_frozen_suite_passes and "
    "not test_mandatory_regression_matrix_passes and "
    "not test_r4_mandatory_regression_matrix"
)

MANDATORY_REGRESSION_SUITES: tuple[tuple[str, list[str]], ...] = (
    NPSC5F_R4_MANDATORY_REGRESSION_SUITES
)


def flatten_regression_targets(
    suites: tuple[tuple[str, list[str]], ...] = MANDATORY_REGRESSION_SUITES,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for _, targets in suites:
        for target in targets:
            if target not in seen:
                seen.add(target)
                ordered.append(target)
    return ordered


def run_mandatory_regression_matrix(
    repo_root: Path,
    *,
    suites: tuple[tuple[str, list[str]], ...] = MANDATORY_REGRESSION_SUITES,
) -> subprocess.CompletedProcess[str]:
    """Run the full matrix in one ``uv run pytest`` invocation."""
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
