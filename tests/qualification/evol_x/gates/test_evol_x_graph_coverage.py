# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.graph.builder import build_semantic_graph
from testing_support.change_safety.graph.coverage import compute_graph_coverage
from testing_support.change_safety.intake.git_change_set import rev_parse_head

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_semantic_graph_digest_stable() -> None:
    head = rev_parse_head()
    inventory = build_closed_world_inventory(source_commit_sha=head)
    graph_a = build_semantic_graph(inventory)
    graph_b = build_semantic_graph(inventory)
    assert graph_a.digest == graph_b.digest
    coverage = compute_graph_coverage(inventory)
    assert coverage.module_nodes > 100
    assert coverage.critical_omissions == 0
    assert 0.0 < coverage.coverage_ratio <= 1.0
