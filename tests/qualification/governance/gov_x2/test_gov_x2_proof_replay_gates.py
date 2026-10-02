# © Artur Czarnecki. All rights reserved.

"""GOV-X2 proof replay gates — inventory integrity without executing the full replay graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.gov_x2.catalog import (
    GOV_X2_E2E_CLASS_NODE_IDS,
    GOV_X2_INVARIANT_CATALOG,
    gov_x2_all_proof_pytest_node_ids,
)
from tests.qualification.governance.gov_x2.proof_replay import (
    build_gov_x2_exact_proof_replay_argv,
    gov_x2_orchestration_pytest_node_prefixes,
    gov_x2_proof_nodes_for_exact_replay,
)
from tests.qualification.governance.pytest_node_integrity import (
    pytest_nodes_missing_from_collection,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_gov_x2_each_invariant_has_at_least_one_proof_node() -> None:
    for entry in GOV_X2_INVARIANT_CATALOG:
        assert entry.primary_pytest_node_ids, entry.invariant_id


def test_gov_x2_proof_inventory_unique_and_deduped() -> None:
    declared: list[str] = []
    for entry in GOV_X2_INVARIANT_CATALOG:
        declared.extend(entry.primary_pytest_node_ids)
    declared.extend(GOV_X2_E2E_CLASS_NODE_IDS)
    unique_declared = tuple(dict.fromkeys(declared))
    inventory = gov_x2_all_proof_pytest_node_ids()
    assert len(inventory) == len(unique_declared)
    assert len(inventory) == len(set(inventory))


def test_gov_x2_orchestration_tests_not_in_proof_inventory() -> None:
    prefixes = gov_x2_orchestration_pytest_node_prefixes()
    for node_id in gov_x2_all_proof_pytest_node_ids():
        assert not any(node_id.startswith(prefix) for prefix in prefixes), node_id


def test_gov_x2_proof_nodes_collectable_for_exact_replay() -> None:
    nodes = gov_x2_proof_nodes_for_exact_replay()
    missing, proc = pytest_nodes_missing_from_collection(nodes, _REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, f"missing={len(missing)}: {sorted(missing)}"


def test_gov_x2_exact_proof_replay_argv_derived_from_inventory() -> None:
    nodes = gov_x2_proof_nodes_for_exact_replay()
    argv = build_gov_x2_exact_proof_replay_argv(nodes)
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
