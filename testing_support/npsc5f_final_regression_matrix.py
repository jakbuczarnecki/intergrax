# © Artur Czarnecki. All rights reserved.

"""NPSC-5F Final — single-invocation mandatory regression matrix (qualification only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES,
)
from testing_support.npsc5f_r4_regression_matrix import flatten_regression_targets

from testing_support.execution_qualification.embedded_harness_kexpr import (
    CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR,
)

# Exclude orchestrators and nested ``uv run pytest`` / ruff / pyright fan-out tests.
_PYTEST_MATRIX_KEXPR = CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR

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
