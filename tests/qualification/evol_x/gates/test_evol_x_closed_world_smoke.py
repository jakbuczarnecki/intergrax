# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.contracts.enums import ImpactStatus, SelectionStatus
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.intake.git_change_set import rev_parse_head
from testing_support.change_safety.selection.selector import select_tests_for_impact
from tests.qualification.evol_x.adapters.qual_x import QualXSuiteCatalogAdapter

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_closed_world_inventory_digest_stable() -> None:
    head = rev_parse_head()
    inv_a = build_closed_world_inventory(source_commit_sha=head)
    inv_b = build_closed_world_inventory(source_commit_sha=head)
    assert inv_a.inventory_digest == inv_b.inventory_digest
    assert len(inv_a.nodes) > 1000
    assert len(inv_a.edges) > 1000


def test_evol_x_impact_and_selection_smoke() -> None:
    head = rev_parse_head()
    inventory = build_closed_world_inventory(source_commit_sha=head)
    impact = assess_path_change_impact(
        inventory,
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
