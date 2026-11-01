# © Artur Czarnecki. All rights reserved.

"""W5-07: EVOL_ADJACENT_REGRESSION (Adj) - explicit COMPAT-X node list is real, collectable and wired into selection."""

from dataclasses import replace

import pytest
from testing_support.change_safety.contracts.enums import SelectionStatus
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.selection.oracle import SelectionFaultClass, inject_selection_fault
from testing_support.change_safety.selection.selector import select_tests
from tests.qualification.evol_x.adapters.qual_x import verify_node_ids_collectable
from tests.qualification.evol_x.adapters.selection_inputs import build_selection_inputs

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_DOC = "evol_adjacent_regression_nodes.v1.json"


def _adjacent() -> frozenset[str]:
    return frozenset(load_evol_x_data_document(_DOC)["node_ids"])


def test_adjacent_registry_is_explicit_nonempty_and_compat_scoped() -> None:
    nodes = _adjacent()
    assert nodes and all("::" in n and n.startswith("tests/qualification/compat_x/") for n in nodes)


def test_adjacent_nodes_are_collectable_in_one_collect_invocation() -> None:
    assert verify_node_ids_collectable(_adjacent()) == frozenset()


def test_adjacent_negative_phantom_node_is_detected() -> None:
    phantom = "tests/qualification/compat_x/test_compat_x_inventory_gates.py::test_does_not_exist"
    assert verify_node_ids_collectable(frozenset({phantom})) == frozenset({phantom})


def test_adjacent_nodes_are_part_of_conservative_universe(evol_inventory) -> None:  # noqa: ANN001
    impact = assess_path_change_impact(
        evol_inventory, changed_paths=frozenset({"testing_support/change_safety/contracts/enums.py"})
    )
    base = build_selection_inputs(impact, expected_head_sha=impact.source_commit_sha)
    assert base.adjacent_node_ids == _adjacent()
    conservative = select_tests(inject_selection_fault(SelectionFaultClass.UNKNOWN_PATH, base))
    assert conservative.status is SelectionStatus.CONSERVATIVE_FULL
    assert _adjacent() <= {n.node_id for n in conservative.selected}


def test_adjacent_negative_uncollectable_adjacent_node_refuses_selection(evol_inventory) -> None:  # noqa: ANN001
    impact = assess_path_change_impact(
        evol_inventory, changed_paths=frozenset({"testing_support/change_safety/contracts/enums.py"})
    )
    base = build_selection_inputs(impact, expected_head_sha=impact.source_commit_sha)
    collectable = (base.catalog_node_ids | base.evol_manifest_node_ids) - _adjacent()
    decision = select_tests(replace(base, collectable_node_ids=frozenset(collectable)))
    assert decision.status is SelectionStatus.REFUSED_INTEGRITY
