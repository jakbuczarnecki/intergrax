# © Artur Czarnecki. All rights reserved.

"""PLUG-03 batch hooks and catalog integrity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.qualification.plug_03.catalog import (
    PLUG_03_EXPLICIT_SELECTION_SURFACES,
    PLUG_03_MAPPED_NODE_IDS,
    PLUG_03_Q4_CHAIN_PARTICIPATION_SURFACES,
    PLUG_03_Q4_PUBLIC_REPLACEMENT_REQUIRED_KINDS,
    PLUG_03_SURFACE_MATRIX,
)

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


def test_plug_03_q4_public_surfaces_have_required_evidence_kinds() -> None:
    for row in PLUG_03_SURFACE_MATRIX:
        if row.level != "Q4":
            continue
        if row.classification != "PUBLIC_EXTERNAL_PLUGIN":
            continue
        kinds = frozenset(kind for ref in row.evidence for kind in ref.kinds)
        assert kinds, f"Q4 row {row.surface!r} must declare evidence kinds"
        required = PLUG_03_Q4_PUBLIC_REPLACEMENT_REQUIRED_KINDS
        if row.surface in PLUG_03_Q4_CHAIN_PARTICIPATION_SURFACES:
            required = required - {"DEFAULT_BYPASS"}
        missing = required - kinds
        assert not missing, f"Q4 row {row.surface!r} missing kinds: {sorted(missing)}"
        assert "CANONICAL_CONSUMPTION" in kinds
        discovery_only = all(
            ref.kinds == ("DISCOVERY",) or ref.kinds == ("DISCOVERY", "ADMISSION")
            for ref in row.evidence
        )
        assert not discovery_only, f"Q4 row {row.surface!r} cannot be discovery-only"
        if row.surface in PLUG_03_EXPLICIT_SELECTION_SURFACES:
            assert "FAIL_CLOSED" in kinds, f"explicit selection surface {row.surface!r} needs FAIL_CLOSED"


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
