# © Artur Czarnecki. All rights reserved.

"""W3-06 / M-04 / S6 precursor: 8 fault classes map to the oracle status; none may bypass to SELECTED."""

from __future__ import annotations

import pytest
from testing_support.change_safety.contracts.enums import SelectionStatus
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.selection.oracle import (
    FAULT_INJECTION_MATRIX,
    SelectionFaultClass,
    expected_status_for_fault,
    inject_selection_fault,
)
from testing_support.change_safety.selection.selector import select_tests
from tests.qualification.evol_x.adapters.selection_inputs import build_selection_inputs

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


@pytest.fixture(scope="module")
def base_inputs(evol_inventory: ClosedWorldInventory):  # noqa: ANN201
    impact = assess_path_change_impact(
        evol_inventory, changed_paths=frozenset({"testing_support/change_safety/contracts/enums.py"})
    )
    inputs = build_selection_inputs(impact, expected_head_sha=impact.source_commit_sha)
    assert select_tests(inputs).status is SelectionStatus.SELECTED  # fault-free baseline is SELECTED
    return inputs


def test_evol_x_selection_fault_matrix_has_eight_classes_with_expected_statuses() -> None:
    assert len(FAULT_INJECTION_MATRIX) >= 8
    assert {r.fault for r in FAULT_INJECTION_MATRIX} == set(SelectionFaultClass)
    assert expected_status_for_fault(SelectionFaultClass.DIGEST_MISMATCH) is SelectionStatus.REFUSED_INTEGRITY
    assert expected_status_for_fault(SelectionFaultClass.UNKNOWN_PATH) is SelectionStatus.CONSERVATIVE_FULL


@pytest.mark.parametrize("fault", list(SelectionFaultClass))
def test_evol_x_selection_fault_injection_matrix(fault: SelectionFaultClass, base_inputs) -> None:  # noqa: ANN001
    decision = select_tests(inject_selection_fault(fault, base_inputs))
    assert decision.status is expected_status_for_fault(fault)
    assert decision.status is not SelectionStatus.SELECTED  # bypass arm: fault must never yield SELECTED
    assert any(r.rule_id == f"FAULT:{fault.value}" for r in decision.reasons)


def test_evol_x_selection_fault_conservative_full_is_scoped_policy_and_superset_of_mandatory(base_inputs) -> None:  # noqa: ANN001
    decision = select_tests(inject_selection_fault(SelectionFaultClass.UNKNOWN_PATH, base_inputs))
    selected_ids = {n.node_id for n in decision.selected}
    assert base_inputs.mandatory_node_ids <= selected_ids
    assert base_inputs.evol_manifest_node_ids <= selected_ids
    assert {n.reason_rule_id for n in decision.selected} == {"EVOL-R7-SCOPED"}
    assert not any("PLATFORM_CONSERVATIVE_FULL" in r.rule_id for r in decision.reasons)
