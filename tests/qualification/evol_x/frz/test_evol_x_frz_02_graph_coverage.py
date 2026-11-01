# © Artur Czarnecki. All rights reserved.

"""FRZ-EVOL-02: graph coverage meets the ratified threshold (EVOL-M-02 >= 90%, M-02C = 0) + CW-06 second method."""

import pytest
from testing_support.change_safety.assess.consumers import critical_consumer_closure
from testing_support.change_safety.discovery.closed_world_gates import run_cw06
from testing_support.change_safety.graph.coverage import compute_graph_coverage

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

M02_THRESHOLD = 0.90


def test_frz_evol_02_text_pairs_are_covered_by_graph_edges(evol_inventory) -> None:  # noqa: ANN001
    closure = critical_consumer_closure(evol_inventory)
    assert closure.text_consumer_pairs > 0, "second-method oracle must find real contract consumers"
    covered = closure.text_consumer_pairs - closure.orphan_count
    assert covered / closure.text_consumer_pairs >= M02_THRESHOLD
    assert closure.orphan_count == 0  # M-02C


def test_frz_evol_02_coverage_report_and_cw06_green(evol_inventory) -> None:  # noqa: ANN001
    report = compute_graph_coverage(evol_inventory)
    assert report.critical_omissions == 0
    assert report.module_nodes > 0
    assert run_cw06(evol_inventory).passed


def test_frz_evol_02_omission_is_detected_when_edge_removed(evol_inventory) -> None:  # noqa: ANN001
    closure = critical_consumer_closure(evol_inventory)
    from testing_support.change_safety.assess.consumers import critical_text_pairs

    src, tgt = critical_text_pairs(evol_inventory)[0]
    trimmed = tuple(
        e
        for e in evol_inventory.edges
        if not (e.source_id == src and e.target_id == tgt and e.rel in {"IMPORTS", "CONSUMES_CONTRACT", "BINDS"})
    )
    broken = evol_inventory.with_edges(trimmed)
    assert critical_consumer_closure(broken).orphan_count > closure.orphan_count
