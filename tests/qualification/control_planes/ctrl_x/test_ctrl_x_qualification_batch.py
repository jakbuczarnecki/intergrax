# © Artur Czarnecki. All rights reserved.

"""CTRL-X — CX-01..CX-12 catalog integrity and proof collectability."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.control_planes.ctrl_x.catalog import (
    CTRL_X_PLANE_CATALOG,
    CTRL_X_START_HEAD,
    CtrlXPlaneResult,
    ctrl_x_all_proof_pytest_node_ids,
)
from tests.qualification.governance.pytest_node_integrity import (
    pytest_nodes_missing_from_collection,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_ctrl_x_catalog_covers_cx_01_through_cx_12() -> None:
    ids = {entry.plane_id for entry in CTRL_X_PLANE_CATALOG}
    expected = {f"CX-{i:02d}" for i in range(1, 13)}
    assert ids == expected


def test_ctrl_x_frz_ctl_criteria_align_one_to_one() -> None:
    for entry in CTRL_X_PLANE_CATALOG:
        num = entry.plane_id.removeprefix("CX-")
        assert entry.frz_criterion == f"FRZ-CTL-{num}"


def test_ctrl_x_planes_declare_pass_with_proof_nodes() -> None:
    for entry in CTRL_X_PLANE_CATALOG:
        assert entry.result is CtrlXPlaneResult.PASS, entry.plane_id
        assert entry.primary_pytest_node_ids, entry.plane_id


def test_ctrl_x_all_proof_pytest_node_ids_are_collectable() -> None:
    combined = ctrl_x_all_proof_pytest_node_ids()
    missing, proc = pytest_nodes_missing_from_collection(combined, _REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, f"missing={len(missing)}: {sorted(missing)}"


def test_ctrl_x_start_head_documented_in_catalog() -> None:
    assert len(CTRL_X_START_HEAD) == 40
