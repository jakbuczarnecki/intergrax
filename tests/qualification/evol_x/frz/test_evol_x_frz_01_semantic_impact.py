# © Artur Czarnecki. All rights reserved.

"""FRZ-EVOL-01: semantic change-impact exists (graph REL traversal, not filename heuristics)."""

import pytest
from testing_support.change_safety.contracts.enums import ImpactStatus
from testing_support.change_safety.graph.impact import assess_path_change_impact

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_CHANGED = "intergrax/contracts/execution_identity.py"


def test_frz_evol_01_impact_is_graph_derived_not_filename_derived(evol_inventory) -> None:  # noqa: ANN001
    impact = assess_path_change_impact(evol_inventory, changed_paths=frozenset({_CHANGED}))
    assert impact.status is ImpactStatus.COMPLETE
    transitive = [n for n in impact.affected if n.kind == "TRANSITIVE" and n.node_id.startswith("module:")]
    assert transitive, "a widely imported identity contract must have transitive consumers"
    assert all((n.rule_id or "").startswith("REL-") for n in transitive)
    unrelated_names = [n for n in transitive if "execution_identity" not in n.node_id]
    assert unrelated_names, "consumers found by REL edges must not depend on filename similarity"
    assert all(len(n.witness_path) >= 2 for n in transitive)  # every consumer carries a witness path


def test_frz_evol_01_unknown_path_is_conservative_never_guessed(evol_inventory) -> None:  # noqa: ANN001
    impact = assess_path_change_impact(evol_inventory, changed_paths=frozenset({"nowhere/unknown.bin"}))
    assert impact.status is ImpactStatus.PARTIAL
    assert [n.kind for n in impact.affected] == ["CONSERVATIVE_UNKNOWN"]


def test_frz_evol_01_impact_digests_bind_to_inventory(evol_inventory) -> None:  # noqa: ANN001
    a = assess_path_change_impact(evol_inventory, changed_paths=frozenset({_CHANGED}))
    b = assess_path_change_impact(evol_inventory, changed_paths=frozenset({_CHANGED}))
    assert a == b and a.inventory_digest == evol_inventory.inventory_digest and a.graph_digest
