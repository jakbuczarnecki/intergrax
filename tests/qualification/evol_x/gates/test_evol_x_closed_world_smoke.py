# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.contracts.enums import ImpactStatus, SelectionStatus
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.selection.selector import select_tests_for_impact
from tests.qualification.evol_x.adapters.qual_x import QualXSuiteCatalogAdapter

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_closed_world_inventory_digest_stable(evol_inventory: ClosedWorldInventory, evol_head: str) -> None:
    from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory

    again = build_closed_world_inventory(source_commit_sha=evol_head)
    assert evol_inventory.inventory_digest == again.inventory_digest
    assert len(evol_inventory.nodes) > 10000
    assert len(evol_inventory.edges) > 50000
    assert evol_inventory.source_tree_id


def test_evol_x_impact_and_selection_smoke(evol_inventory: ClosedWorldInventory) -> None:
    impact = assess_path_change_impact(
        evol_inventory,
        changed_paths=frozenset({"testing_support/change_safety/contracts/enums.py"}),
    )
    assert impact.status in {ImpactStatus.COMPLETE, ImpactStatus.PARTIAL}
    catalog = QualXSuiteCatalogAdapter()
    decision = select_tests_for_impact(
        impact,
        catalog_node_ids=catalog.collectable_node_ids(),
        mandatory_node_ids=catalog.mandatory_node_ids(),
    )
    assert decision.status in {SelectionStatus.SELECTED, SelectionStatus.CONSERVATIVE_FULL}
    assert decision.mandatory or decision.selected
