# © Artur Czarnecki. All rights reserved.

"""B-M03-02: independent O-F causal audit vs selector (P-S arcs); artifact @ evidence build."""

from __future__ import annotations

from typing import Any

from testing_support.change_safety.contracts.enums import SelectionStatus
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.selection.selector import select_tests
from tests.qualification.evol_x.adapters.selection_inputs import build_selection_inputs
from tests.qualification.evol_x.metrics.measure import _SELF_SURFACE, _reverse_closure


def _expected_selected_nodes(inventory: ClosedWorldInventory, changed: frozenset[str], head: str) -> frozenset[str]:
    impact = assess_path_change_impact(inventory, changed_paths=changed)
    base = build_selection_inputs(impact, expected_head_sha=head)
    closure = _reverse_closure(changed)
    touches_self = any(p.startswith(_SELF_SURFACE) for p in closure)
    universe = base.mandatory_node_ids | base.pop_a_node_ids | base.evol_manifest_node_ids | base.adjacent_node_ids
    return frozenset(
        n
        for n in universe
        if n in base.mandatory_node_ids
        or n.split("::", 1)[0] in closure
        or (touches_self and n in base.evol_manifest_node_ids)
    )


def build_m03_causal_audit(inventory: ClosedWorldInventory, head: str) -> dict[str, Any]:
    oracle_doc = load_evol_x_data_document("m03_oracle_declared.v1.json")
    declared_raw = {row["arc_id"]: row.get("expected_consumers") for row in oracle_doc["arcs"]}
    rows: list[dict[str, Any]] = []
    for arc in load_evol_x_data_document("metric_populations.v1.json")["p_s_arcs"]:
        arc_id = arc["arc_id"]
        changed = frozenset(arc["changed_paths"])
        impact = assess_path_change_impact(inventory, changed_paths=changed)
        base = build_selection_inputs(impact, expected_head_sha=head)
        decision = select_tests(base)
        selected = frozenset(n.node_id for n in decision.selected) if decision.selected else frozenset()
        expected = _expected_selected_nodes(inventory, changed, head)
        raw = declared_raw.get(arc_id)
        oracle_expected = expected if raw is None else frozenset(raw)
        causal_ok = oracle_expected == expected and (decision.status is SelectionStatus.SELECTED or not expected)
        selector_covers_oracle = oracle_expected <= selected if decision.status is SelectionStatus.SELECTED else False
        rows.append(
            {
                "arc_id": arc_id,
                "oracle_expected_count": len(oracle_expected),
                "independent_expected_count": len(expected),
                "selected_count": len(selected),
                "oracle_matches_independent": oracle_expected == expected,
                "selector_covers_oracle": selector_covers_oracle,
                "passed": causal_ok and selector_covers_oracle,
            }
        )
    return {
        "schema_version": "m03_causal_audit.v1",
        "source_commit_sha": head,
        "passed": all(r["passed"] for r in rows),
        "rows": rows,
    }


def verify_causal_audit_against_oracle(audit: dict[str, Any], oracle_doc: dict[str, Any]) -> list[str]:
    declared_raw = {row["arc_id"]: row.get("expected_consumers") for row in oracle_doc["arcs"]}
    problems: list[str] = []
    for row in audit.get("rows", ()):
        arc_id = row["arc_id"]
        raw = declared_raw.get(arc_id)
        if raw is None:
            continue
        if not row.get("oracle_matches_independent"):
            problems.append(f"causal audit independent set diverged from oracle for {arc_id}")
    if not audit.get("passed"):
        problems.append("m03_causal_audit.passed is false")
    return problems
