# © Artur Czarnecki. All rights reserved.

"""SESSION-01 — single reproducible qualification batch (existing test suites only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.qualification.session_01.catalog import (
    SESSION_01_Q_CATALOG,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

SESSION_01_CROSS_WORKER_TARGET = (
    "tests/integration/applications/test_unified_execution_entry_j3.py::"
    "test_worker_checkpoint_resume_via_queue_payload"
)

SESSION_01_MAPPED_NODE_IDS: tuple[str, ...] = tuple(
    node_id for entry in SESSION_01_Q_CATALOG for node_id in entry.pytest_node_ids
)


def test_session_01_q_evidence_node_ids_collect() -> None:
    node_ids = [node for entry in SESSION_01_Q_CATALOG for node in entry.pytest_node_ids]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *node_ids],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_session_01_full_qualification_batch() -> None:
    """One reproducible SESSION-Q1..Q20 batch (mapped node ids + cross-worker proof)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *SESSION_01_MAPPED_NODE_IDS,
            SESSION_01_CROSS_WORKER_TARGET,
            "-q",
            "--tb=short",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
