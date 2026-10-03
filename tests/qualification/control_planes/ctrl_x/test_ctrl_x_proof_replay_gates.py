# © Artur Czarnecki. All rights reserved.

"""CTRL-X proof replay gates — inventory integrity without recursive replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.control_planes.ctrl_x.catalog import (
    CTRL_X_CROSS_CUTTING_PROOF_NODE_IDS,
    CTRL_X_PLANE_CATALOG,
    ctrl_x_all_proof_pytest_node_ids,
)
from tests.qualification.control_planes.ctrl_x.proof_replay import (
    build_ctrl_x_exact_proof_replay_argv,
    ctrl_x_orchestration_pytest_node_prefixes,
    ctrl_x_proof_nodes_for_exact_replay,
)
from tests.qualification.governance.pytest_node_integrity import (
    pytest_nodes_missing_from_collection,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_ctrl_x_each_plane_has_at_least_one_proof_node() -> None:
    for entry in CTRL_X_PLANE_CATALOG:
        assert entry.primary_pytest_node_ids, entry.plane_id


def test_ctrl_x_proof_inventory_unique_and_deduped() -> None:
    declared: list[str] = []
    for entry in CTRL_X_PLANE_CATALOG:
        declared.extend(entry.primary_pytest_node_ids)
    declared.extend(CTRL_X_CROSS_CUTTING_PROOF_NODE_IDS)
    unique_declared = tuple(dict.fromkeys(declared))
    inventory = ctrl_x_all_proof_pytest_node_ids()
    assert len(inventory) == len(unique_declared)
    assert len(inventory) == len(set(inventory))


def test_ctrl_x_orchestration_tests_not_in_proof_inventory() -> None:
    prefixes = ctrl_x_orchestration_pytest_node_prefixes()
    for node_id in ctrl_x_all_proof_pytest_node_ids():
        assert not any(node_id.startswith(prefix) for prefix in prefixes), node_id


def test_ctrl_x_proof_nodes_collectable_for_exact_replay() -> None:
    nodes = ctrl_x_proof_nodes_for_exact_replay()
    missing, proc = pytest_nodes_missing_from_collection(nodes, _REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, f"missing={len(missing)}: {sorted(missing)}"


def test_ctrl_x_exact_proof_replay_argv_derived_from_inventory() -> None:
    nodes = ctrl_x_proof_nodes_for_exact_replay()
    argv = build_ctrl_x_exact_proof_replay_argv(nodes)
    assert argv[:6] == [
        argv[0],
        "-m",
        "pytest",
        "-p",
        "no:xdist",
        "-q",
    ]
    assert argv[6] == "--tb=short"
    assert tuple(argv[7:]) == nodes
