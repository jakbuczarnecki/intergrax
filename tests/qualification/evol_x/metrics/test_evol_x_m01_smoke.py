# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.contracts.enums import ImpactStatus
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.graph.impact import assess_path_change_impact

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_m01_complete_classification_on_known_path(evol_inventory: ClosedWorldInventory) -> None:
    impact = assess_path_change_impact(
        evol_inventory,
        changed_paths=frozenset({"intergrax/__init__.py"}),
    )
    assert impact.status is ImpactStatus.COMPLETE
    assert impact.affected
