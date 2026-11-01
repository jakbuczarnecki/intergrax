# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.assess.consumers import build_consumer_index, orphan_import_targets
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.intake.git_change_set import rev_parse_head

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_consumer_index_non_empty() -> None:
    head = rev_parse_head()
    inventory = build_closed_world_inventory(source_commit_sha=head)
    index = build_consumer_index(inventory)
    assert index.consumers_by_target
    module_ids = frozenset(n.node_id for n in inventory.nodes if n.node_id.startswith("module:"))
    orphans = orphan_import_targets(inventory.edges, module_ids)
    assert isinstance(orphans, tuple)
