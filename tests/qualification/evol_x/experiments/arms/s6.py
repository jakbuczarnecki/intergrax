# © Artur Czarnecki. All rights reserved.

"""EXP-S6: negative regression - every constructed fault is detected; no bypass to SELECTED (M-05)."""

from __future__ import annotations

from testing_support.change_safety.contracts.enums import SelectionStatus
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.selection.oracle import (
    SelectionFaultClass,
    expected_status_for_fault,
    inject_selection_fault,
)
from testing_support.change_safety.selection.selector import select_tests
from tests.qualification.evol_x.adapters.selection_inputs import build_selection_inputs
from tests.qualification.evol_x.experiments._harness import ArmResult, load_oracle

SCENARIO = "S6"


def run(inventory: ClosedWorldInventory) -> tuple[ArmResult, ArmResult]:
    changed = frozenset(load_oracle("s6_negative_regression")["changed_paths"])
    impact = assess_path_change_impact(inventory, changed_paths=changed)
    base = build_selection_inputs(impact, expected_head_sha=impact.source_commit_sha)
    baseline = select_tests(base)
    detected = 0
    statuses: dict[str, str] = {}
    for fault in SelectionFaultClass:
        decision = select_tests(inject_selection_fault(fault, base))
        statuses[fault.value] = decision.status.value
        if decision.status is expected_status_for_fault(fault) and any(
            r.rule_id == f"FAULT:{fault.value}" for r in decision.reasons
        ):
            detected += 1
    total = len(SelectionFaultClass)
    pos_ok = baseline.status is SelectionStatus.SELECTED and detected == total
    # Bypass arm: compound faults and fault+clean-analyzer combinations must never produce SELECTED.
    bypass = 0
    for first in SelectionFaultClass:
        for second in SelectionFaultClass:
            compound = inject_selection_fault(second, inject_selection_fault(first, base))
            if select_tests(compound).status is SelectionStatus.SELECTED:
                bypass += 1
    neg_ok = bypass == 0
    return (
        ArmResult(SCENARIO, "positive", pos_ok, f"detected={detected}/{total} baseline={baseline.status.value}"),
        ArmResult(SCENARIO, "negative", neg_ok, f"bypass_count={bypass} of {total * total} compound injections"),
    )
