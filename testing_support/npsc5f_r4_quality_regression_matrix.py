# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R4 Final reconstruction quality — mandatory regression matrix (qualification only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

_PYTEST_MATRIX_KEXPR = (
    "not test_mandatory_frozen_suite_passes and "
    "not test_mandatory_regression_matrix_passes and "
    "not test_r4_quality_mandatory_regression_matrix"
)

MANDATORY_REGRESSION_SUITES: tuple[tuple[str, list[str]], ...] = (
    (
        "R4 reconstruction quality gate",
        ["tests/unit/runtime/architecture/test_npsc5f_r4_reconstruction_quality.py"],
    ),
    (
        "Execution reconstruction",
        ["tests/unit/runtime/diagnostics/test_execution_reconstruction.py"],
    ),
    (
        "R3 Final",
        ["tests/unit/runtime/architecture/test_npsc5f_r3_final_governed_evidence_export.py"],
    ),
    (
        "R2 Final",
        ["tests/unit/runtime/architecture/test_npsc5f_r2_final_journal_completeness_ordering.py"],
    ),
    (
        "R1 Final",
        ["tests/unit/runtime/architecture/test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py"],
    ),
    (
        "R4 quality Final drift sentinel",
        ["tests/unit/testing_support/test_npsc5f_r4_quality_protected_drift.py"],
    ),
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
