# © Artur Czarnecki. All rights reserved.

"""PLUG-03 batch hooks and catalog integrity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.qualification.plug_03.catalog import PLUG_03_MAPPED_NODE_IDS, PLUG_03_SURFACE_MATRIX

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

PLUG_03_BATCH_TARGET_NODE_IDS: tuple[str, ...] = tuple(
    node_id
    for node_id in PLUG_03_MAPPED_NODE_IDS
    if "test_plug_03_full_qualification_evidence_batch" not in node_id
)


def test_plug_03_matrix_covers_required_public_surfaces() -> None:
    surfaces = {row.surface for row in PLUG_03_SURFACE_MATRIX}
    required = {
        "Tools / ToolPlugin",
        "ToolInvocationPattern",
        "Skills / SkillPlugin",
        "Integrations",
        "Nexus encapsulation gate",
    }
    assert required.issubset(surfaces)


def test_plug_03_catalog_pytest_nodes_resolve() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *PLUG_03_BATCH_TARGET_NODE_IDS],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_plug_03_full_qualification_evidence_batch() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *PLUG_03_BATCH_TARGET_NODE_IDS, "-q", "--tb=short"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
