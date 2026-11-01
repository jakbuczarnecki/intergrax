# © Artur Czarnecki. All rights reserved.

"""W2-06/W2-07: semantic graph digest (REL-aware, XG-03) + real critical_omissions oracle (XG-02)."""

from __future__ import annotations

import pytest
from testing_support.change_safety.assess.consumers import critical_consumer_closure, critical_text_pairs
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.discovery.extractors import GraphEdge
from testing_support.change_safety.graph.builder import build_semantic_graph
from testing_support.change_safety.graph.coverage import compute_graph_coverage

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_semantic_graph_digest_stable(evol_inventory: ClosedWorldInventory) -> None:
    graph_a = build_semantic_graph(evol_inventory)
    graph_b = build_semantic_graph(evol_inventory)
    assert graph_a.digest == graph_b.digest
    assert graph_a.digest != graph_a.inventory_digest  # XG-03: distinct, but bound to inventory
    coverage = compute_graph_coverage(evol_inventory)
    assert coverage.module_nodes > 1000
    assert coverage.critical_omissions == 0
    assert 0.0 < coverage.coverage_ratio <= 1.0
    assert 0.0 < coverage.import_ratio_internal < 1.0  # XG-01 diagnostic


def test_evol_x_graph_digest_is_rel_aware_and_tamper_evident(evol_inventory: ClosedWorldInventory) -> None:
    base = build_semantic_graph(evol_inventory)
    first = evol_inventory.edges[0]
    retyped = (GraphEdge(first.source_id, first.target_id, "BINDS"),) + evol_inventory.edges[1:]
    assert build_semantic_graph(evol_inventory.with_edges(retyped)).digest != base.digest
    dropped = evol_inventory.with_edges(evol_inventory.edges[1:])
    assert build_semantic_graph(dropped).digest != base.digest


def test_evol_x_critical_omissions_oracle_is_real_not_hardcoded(evol_inventory: ClosedWorldInventory) -> None:
    baseline = critical_consumer_closure(evol_inventory)
    assert baseline.text_consumer_pairs > 100
    assert baseline.omissions == ()
    victim_src, victim_tgt = critical_text_pairs(evol_inventory)[0]
    pruned = evol_inventory.with_edges(
        tuple(e for e in evol_inventory.edges if not (e.source_id == victim_src and e.target_id == victim_tgt))
    )
    after = critical_consumer_closure(pruned)
    assert f"{victim_src}->{victim_tgt}" in after.omissions
    assert compute_graph_coverage(pruned).critical_omissions >= 1


def test_evol_x_leave_one_out_each_sampled_consumer_detected(evol_inventory: ClosedWorldInventory) -> None:
    pairs = critical_text_pairs(evol_inventory)
    sample = pairs[:: max(1, len(pairs) // 8)][:8]
    for src, tgt in sample:
        pruned = evol_inventory.with_edges(
            tuple(e for e in evol_inventory.edges if not (e.source_id == src and e.target_id == tgt))
        )
        assert f"{src}->{tgt}" in critical_consumer_closure(pruned).omissions
