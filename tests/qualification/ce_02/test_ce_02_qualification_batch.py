# © Artur Czarnecki. All rights reserved.

"""CE-02 batch hooks and catalog integrity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.qualification.ce_02.catalog import CE_02_Q_CATALOG

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

CE_02_MAPPED_NODE_IDS: tuple[str, ...] = tuple(
    node_id for entry in CE_02_Q_CATALOG for node_id in entry.pytest_node_ids
)

# CE2-Q18 maps to this batch hook; exclude it to avoid recursive re-entry.
CE_02_BATCH_TARGET_NODE_IDS: tuple[str, ...] = tuple(
    node_id
    for node_id in CE_02_MAPPED_NODE_IDS
    if "test_ce_02_full_qualification_evidence_batch" not in node_id
)


def test_ce_02_catalog_covers_ce2_q1_through_ce2_q18() -> None:
    ids = {entry.q_id for entry in CE_02_Q_CATALOG}
    expected = {f"CE2-Q{i}" for i in range(1, 19)}
    assert ids == expected


def test_ce_02_catalog_pytest_nodes_resolve() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *CE_02_BATCH_TARGET_NODE_IDS],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ce_02_full_qualification_evidence_batch() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *CE_02_BATCH_TARGET_NODE_IDS, "-q", "--tb=short"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
