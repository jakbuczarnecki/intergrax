# © Artur Czarnecki. All rights reserved.

"""EVOL-M-09H cause identity oracle — registry ↔ diagnostic derivation gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.diagnostics.m09_cause_identity import (
    assess_seed_path_impact,
    cause_identity_digest_for_node,
    describe_attribution_evidence,
    direct_seed_cause_node,
    expected_cause_identities_for_seed,
)
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.evidence import p03
from testing_support.change_safety.graph.impact import assess_path_change_impact

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[4]
_REGISTRY = _REPO / p03.BUNDLE_REL / p03.M09_DRILL_CASE_REGISTRY_NAME
_SEED_PATHS = (
    "intergrax/contracts/execution_identity.py",
    "testing_support/change_safety/graph/impact.py",
    "testing_support/change_safety/contracts/enums.py",
)


def _head() -> str:
    import subprocess

    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


@pytest.fixture(scope="module")
def inventory():
    return build_closed_world_inventory(source_commit_sha=_head())


def test_committed_registry_expected_matches_oracle(inventory) -> None:  # noqa: ANN001
    raw = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    for entry in raw["cases"]:
        derived = expected_cause_identities_for_seed(inventory, entry["changed_path"])
        assert tuple(entry["expected_cause_identities"]) == derived


@pytest.mark.parametrize(
    ("case_id", "path"),
    [("C1", _SEED_PATHS[0]), ("C2", _SEED_PATHS[1]), ("C3", _SEED_PATHS[2])],
    ids=["C1", "C2", "C3"],
)
def test_seed_cause_identity_matrix(inventory, case_id: str, path: str) -> None:  # noqa: ANN001
    impact = assess_seed_path_impact(inventory, path)
    node = direct_seed_cause_node(impact, path)
    digest = cause_identity_digest_for_node(node)
    evidence = describe_attribution_evidence(impact, path)
    assert evidence["attributed_node_id"] == node.node_id
    oracle = expected_cause_identities_for_seed(inventory, path)
    assert len(oracle) == 1
    assert oracle[0] == digest


def test_wrong_node_kind_changes_digest(inventory) -> None:  # noqa: ANN001
    path = _SEED_PATHS[1]
    impact = assess_seed_path_impact(inventory, path)
    node = direct_seed_cause_node(impact, path)
    wrong = stable_digest_hex(["TRANSITIVE", node.node_id])
    assert wrong != cause_identity_digest_for_node(node)


def test_unknown_seed_path_fails_closed(inventory) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="exactly one DIRECT"):
        expected_cause_identities_for_seed(inventory, "intergrax/no/such/module.py")


def test_impact_without_direct_module_fails(inventory) -> None:  # noqa: ANN001
    impact = assess_path_change_impact(inventory, changed_paths=frozenset({"README.md"}))
    with pytest.raises(ValueError, match="exactly one DIRECT"):
        direct_seed_cause_node(impact, "README.md")


def test_attribution_evidence_is_sufficient_to_compute_identity(inventory) -> None:  # noqa: ANN001
    """Evaluator path: read kind + node_id from evidence, apply canonical digest."""
    path = _SEED_PATHS[0]
    impact = assess_seed_path_impact(inventory, path)
    evidence = describe_attribution_evidence(impact, path)
    recomputed = stable_digest_hex([evidence["attributed_node_kind"], evidence["attributed_node_id"]])
    assert recomputed == expected_cause_identities_for_seed(inventory, path)[0]
