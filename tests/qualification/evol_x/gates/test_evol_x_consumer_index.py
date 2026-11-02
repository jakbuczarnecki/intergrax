# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.assess.consumers import (
    build_consumer_index,
    critical_consumer_closure,
    orphan_import_targets,
)
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_consumer_index_non_empty(evol_inventory: ClosedWorldInventory) -> None:
    index = build_consumer_index(evol_inventory)
    assert index.consumers_by_target
    orphans = orphan_import_targets(evol_inventory.edges, evol_inventory.module_ids)
    assert orphans == ()


def test_evol_x_consumer_closure_orphans_zero_on_head(evol_inventory: ClosedWorldInventory) -> None:
    report = critical_consumer_closure(evol_inventory)
    assert report.schema_version == "evol_consumer_closure.v1"
    assert report.orphan_count == 0
    assert report.critical_targets > 5


def test_evol_x_orphan_import_targets_detects_dangling() -> None:
    from testing_support.change_safety.discovery.extractors import GraphEdge

    edges = (GraphEdge("module:a", "module:missing", "IMPORTS"),)
    assert orphan_import_targets(edges, frozenset({"module:a"})) == ("module:missing",)


def test_evol_x_w3_02_hidden_consumer_detected_and_orphans_zero_on_head(evol_inventory: ClosedWorldInventory) -> None:
    from testing_support.change_safety.assess.consumers import (
        consumers_of_targets,
        critical_text_pairs,
        hidden_consumers_of_targets,
    )

    pairs = critical_text_pairs(evol_inventory)
    targets = frozenset(t for _s, t in pairs)
    assert hidden_consumers_of_targets(evol_inventory, targets) == ()  # orphan == 0
    src, tgt = pairs[0]
    assert src in consumers_of_targets(evol_inventory, frozenset({tgt}))
    pruned = evol_inventory.with_edges(
        tuple(e for e in evol_inventory.edges if not (e.source_id == src and e.target_id == tgt))
    )
    assert src in hidden_consumers_of_targets(pruned, frozenset({tgt}))  # orphan > 0 arm
