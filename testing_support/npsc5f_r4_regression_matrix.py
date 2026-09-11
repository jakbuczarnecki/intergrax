# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R4 Final — single-invocation mandatory regression matrix targets (qualification only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

# Exclude orchestrator tests that spawn nested ``uv run pytest`` (process explosion / recursion).
_PYTEST_MATRIX_KEXPR = (
    "not test_mandatory_frozen_suite_passes and "
    "not test_mandatory_regression_matrix_passes and "
    "not test_r4_mandatory_regression_matrix"
)

MANDATORY_REGRESSION_SUITES: tuple[tuple[str, list[str]], ...] = (
    (
        "R4 implementation gate",
        ["tests/unit/runtime/architecture/test_npsc5f_r4_reconstruction_asof_bitemporal.py"],
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
        "NPSC-5F P0",
        ["tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py"],
    ),
    (
        "TRACE-ASOF",
        [
            "tests/unit/runtime/events/test_execution_position_asof.py",
            "tests/unit/runtime/events/test_asof_projection.py",
        ],
    ),
    (
        "TRACE-BITEMP",
        [
            "tests/unit/contracts/test_bitemporal_revision_ordering.py",
            "tests/unit/contracts/test_bitemporal_knowledge.py",
            "tests/unit/runtime/observability/test_knowledge_reconstruction.py",
        ],
    ),
    (
        "Execution reconstruction",
        ["tests/unit/runtime/diagnostics/test_execution_reconstruction.py"],
    ),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/",
        ],
    ),
    (
        "NPSC-5E Final",
        ["tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"],
    ),
    (
        "NPSC-5D Final",
        ["tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"],
    ),
    (
        "NPSC-5C",
        [
            "tests/unit/runtime/architecture/test_npsc5c_coordination_intent_gate.py",
            "tests/unit/runtime/architecture/test_npsc5c_decision_projection_gate.py",
        ],
    ),
    (
        "NPSC-5B Final",
        ["tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py"],
    ),
    (
        "NPSC-5A",
        ["tests/unit/runtime/architecture/test_npsc5a_multi_agent_coordination_gate.py"],
    ),
    (
        "R4 Final drift sentinel",
        ["tests/unit/testing_support/test_npsc5f_r4_final_protected_drift.py"],
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
