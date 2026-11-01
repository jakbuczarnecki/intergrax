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
