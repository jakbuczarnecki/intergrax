# © Artur Czarnecki. All rights reserved.

"""CE-01 batch hooks and catalog integrity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.qualification.ce_01.catalog import CE_01_Q_CATALOG

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

CE_01_MAPPED_NODE_IDS: tuple[str, ...] = tuple(
    node_id for entry in CE_01_Q_CATALOG for node_id in entry.pytest_node_ids
)


def test_ce_01_catalog_covers_ce_q1_through_ce_q15() -> None:
    ids = {entry.q_id for entry in CE_01_Q_CATALOG}
    expected = {f"CE-Q{i}" for i in range(1, 16)}
    assert ids == expected


def test_ce_01_catalog_pytest_nodes_resolve() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *CE_01_MAPPED_NODE_IDS],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ce_01_full_qualification_evidence_batch() -> None:
    """Execute every catalog evidence node (CE-Q1..CE-Q15 mapped tests)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *CE_01_MAPPED_NODE_IDS,
            "-q",
            "--tb=short",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ce_01_frozen_context_convergence_regression_paths() -> None:
    paths = (
        "tests/unit/runtime/nexus/context/test_mem_xint4_single_context_composition.py",
        "tests/unit/context/test_mem_xint5r_authority_and_policy_replaceability.py",
        "tests/unit/context/test_mem_xint6r_typed_source_boundary.py",
        "tests/integration/runtime/test_context_engine_paths.py",
        "tests/unit/context/test_context_tier0_import_boundary.py",
        "scripts/maintenance/check_context_tier0_import_boundary.py",
    )
    for path in paths:
        assert path.startswith("tests/") or path.startswith("scripts/")
