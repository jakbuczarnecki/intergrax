# © Artur Czarnecki. All rights reserved.

"""W3-04/W3-05: EVOL_CONSERVATIVE_FULL union (T1∪T2∪T3∪Adj \\ exclusions), integrity refusal, deterministic ids."""

from __future__ import annotations

from dataclasses import replace

import pytest
from testing_support.change_safety.contracts.enums import ImpactStatus, SelectionStatus
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.selection.policy import (
    ConservativeExclusions,
    evol_conservative_full_set,
    load_conservative_exclusions,
)
from testing_support.change_safety.selection.selector import select_tests
from tests.qualification.evol_x.adapters.qual_x import QualXSuiteCatalogAdapter, verify_node_ids_collectable
from tests.qualification.evol_x.adapters.selection_inputs import build_selection_inputs

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


@pytest.fixture(scope="module")
def self_impact(evol_inventory: ClosedWorldInventory):  # noqa: ANN201
    return assess_path_change_impact(
        evol_inventory, changed_paths=frozenset({"testing_support/change_safety/contracts/enums.py"})
    )


def test_evol_x_t1_t2_node_ids_are_collectable_w3_04() -> None:
    catalog = QualXSuiteCatalogAdapter()
    t1, t2 = catalog.mandatory_min_node_ids(), catalog.pop_a_collectable_node_ids()
    assert t1 and t1 <= t2 <= catalog.collectable_node_ids()
    assert verify_node_ids_collectable(t2) == frozenset()


def test_evol_x_t_missing_collect_detected_w3_04() -> None:
    bogus = frozenset({"tests/qualification/qual_x/test_does_not_exist_at_all.py::test_x"})
    assert verify_node_ids_collectable(bogus) == bogus


def test_evol_x_conservative_full_is_union_minus_exclusions() -> None:
    ex = load_conservative_exclusions()
    assert "DIM-B-PROD-Q-LIVE" in ex.populations
    assert ex.excludes("tests/unit/agents/test_reference_harness_strict.py::test_x")
    t1, t2, t3 = (
        frozenset({"a.py::1"}),
        frozenset({"b.py::2"}),
        frozenset({"c.py::3", "tests/unit/agents/test_reference_harness_strict.py::t"}),
    )
    full = evol_conservative_full_set(t1=t1, t2=t2, t3=t3, adjacent=frozenset({"d.py::4"}), exclusions=ex)
    assert full == {"a.py::1", "b.py::2", "c.py::3", "d.py::4"}


def test_evol_x_dim_b_nodes_never_in_pop_a_selection(self_impact) -> None:  # noqa: ANN001
    catalog = QualXSuiteCatalogAdapter()
    dim_b = catalog.dim_b_excluded_modules()
    inputs = build_selection_inputs(self_impact)
    decision = select_tests(replace(inputs, pop_a_node_ids=inputs.pop_a_node_ids | {f"{next(iter(dim_b))}::test_x"}))
    assert not any(n.node_id.split("::", 1)[0] in dim_b for n in decision.selected)


def test_evol_x_selected_contains_t1_and_t3_when_self_surface_changes_and_policy_label(self_impact) -> None:  # noqa: ANN001
    inputs = build_selection_inputs(self_impact)
    decision = select_tests(inputs)
    assert decision.status is SelectionStatus.SELECTED
    selected = {n.node_id: n.reason_rule_id for n in decision.selected}
    assert inputs.mandatory_node_ids <= set(selected)
    assert inputs.evol_manifest_node_ids <= set(selected)
    assert {"R2-MANDATORY", "R3-EVOL-VL3"} <= set(selected.values())
    assert decision.mandatory_manifest_digest and decision.suite_collection_digest


def test_evol_x_shrunk_manifest_detected_as_under_selection(self_impact) -> None:  # noqa: ANN001
    full = build_selection_inputs(self_impact)
    shrunk = replace(full, evol_manifest_node_ids=frozenset(sorted(full.evol_manifest_node_ids)[:3]))
    full_ids = {n.node_id for n in select_tests(full).selected}
    shrunk_ids = {n.node_id for n in select_tests(shrunk).selected}
    assert shrunk_ids < full_ids  # shrinking the manifest strictly shrinks selection: the pinned digest guards this
    assert select_tests(full).mandatory_manifest_digest == select_tests(shrunk).mandatory_manifest_digest
    from testing_support.change_safety.selection.ci_manifest import manifest_content_digest

    assert manifest_content_digest(sorted(full.evol_manifest_node_ids)) != manifest_content_digest(
        sorted(shrunk.evol_manifest_node_ids)
    )


def test_evol_x_selection_surface_rule_maps_affected_suite_to_catalog_node(
    evol_inventory: ClosedWorldInventory,
) -> None:
    catalog = QualXSuiteCatalogAdapter()
    t2_modules = {n.split("::", 1)[0] for n in catalog.pop_a_collectable_node_ids()}
    candidates = [
        (mod, suites)
        for mod, suites in evol_inventory.tested_by.items()
        if any(s[len("suite:") :] in t2_modules for s in suites)
    ]
    if not candidates:
        pytest.skip("no code module imported by a QUAL-X Pop-A suite at HEAD")
    module_id, _ = sorted(candidates)[0]
    path = next(n.path for n in evol_inventory.nodes if n.node_id == module_id)
    impact = assess_path_change_impact(evol_inventory, changed_paths=frozenset({path}))
    decision = select_tests(build_selection_inputs(impact))
    assert decision.status is SelectionStatus.SELECTED
    surface_reasons = [r for r in decision.reasons if r.rule_id == "R1-SURFACE"]
    assert surface_reasons and all(r.witness for r in surface_reasons)
    assert {r.node_id for r in surface_reasons} <= {n.node_id for n in decision.selected}


def test_evol_x_decision_id_is_deterministic_digest_not_uuid(self_impact) -> None:  # noqa: ANN001
    inputs = build_selection_inputs(self_impact)
    a, b = select_tests(inputs), select_tests(inputs)
    assert a.decision_id == b.decision_id and len(a.decision_id) == 32
    other = select_tests(replace(inputs, mandatory_node_ids=inputs.mandatory_node_ids | {"tests/x.py::t"}))
    assert other.decision_id != a.decision_id


def test_evol_x_uncollectable_required_node_refuses_integrity(self_impact) -> None:  # noqa: ANN001
    inputs = build_selection_inputs(self_impact)
    all_required = (
        inputs.mandatory_node_ids | inputs.pop_a_node_ids | inputs.evol_manifest_node_ids | inputs.adjacent_node_ids
    )
    ok = select_tests(replace(inputs, collectable_node_ids=all_required))
    assert ok.status is SelectionStatus.SELECTED
    missing_one = sorted(all_required)[0]
    refused = select_tests(replace(inputs, collectable_node_ids=all_required - {missing_one}))
    assert refused.status is SelectionStatus.REFUSED_INTEGRITY


def test_evol_x_rc7_excluded_node_does_not_block_collect_integrity(self_impact) -> None:  # noqa: ANN001
    inputs = build_selection_inputs(self_impact)
    rc7 = "tests/unit/agents/test_reference_harness_strict.py::test_x"
    with_rc7 = replace(inputs, evol_manifest_node_ids=inputs.evol_manifest_node_ids | {rc7})
    required = (
        inputs.mandatory_node_ids | inputs.pop_a_node_ids | inputs.evol_manifest_node_ids | inputs.adjacent_node_ids
    )
    decision = select_tests(replace(with_rc7, collectable_node_ids=required))
    assert decision.status is SelectionStatus.SELECTED  # RC-7 excluded (OD-EVOL-10), not required collectable
    assert rc7 not in {n.node_id for n in decision.selected}


def test_evol_x_critical_omissions_never_pass_on_profile_alone(self_impact) -> None:  # noqa: ANN001
    decision = select_tests(replace(build_selection_inputs(self_impact), critical_omissions=3))
    assert decision.status is SelectionStatus.CONSERVATIVE_FULL


def test_evol_x_empty_impact_selects_mandatory_only_and_unknown_is_conservative(
    evol_inventory: ClosedWorldInventory,
) -> None:
    empty = assess_path_change_impact(evol_inventory, changed_paths=frozenset())
    assert empty.status is ImpactStatus.COMPLETE
    decision = select_tests(build_selection_inputs(empty))
    assert decision.status is SelectionStatus.SELECTED
    assert {n.node_id for n in decision.selected} == set(build_selection_inputs(empty).mandatory_node_ids)
    unknown = assess_path_change_impact(evol_inventory, changed_paths=frozenset({"intergrax/mystery/blob.bin"}))
    assert select_tests(build_selection_inputs(unknown)).status is SelectionStatus.CONSERVATIVE_FULL


def test_evol_x_exclusions_type_is_frozen_policy_object() -> None:
    assert isinstance(load_conservative_exclusions(), ConservativeExclusions)
