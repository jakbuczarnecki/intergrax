# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.contracts.enums import ImpactStatus
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.intake.git_change_set import rev_parse_head

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_m01_complete_classification_on_known_path() -> None:
    head = rev_parse_head()
    inventory = build_closed_world_inventory(source_commit_sha=head)
    impact = assess_path_change_impact(
        inventory,
        changed_paths=frozenset({"intergrax/__init__.py"}),
    )
    assert impact.status is ImpactStatus.COMPLETE
    assert impact.affected
