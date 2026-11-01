# © Artur Czarnecki. All rights reserved.

"""B-M03-01/02: fault injection safety + O-F causal audit artifact."""

import copy

import pytest
from testing_support.change_safety.contracts.enums import SelectionStatus
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.selection.oracle import FAULT_INJECTION_MATRIX, inject_selection_fault
from testing_support.change_safety.selection.selector import select_tests
from tests.qualification.evol_x.adapters.selection_inputs import build_selection_inputs
from tests.qualification.evol_x.experiments._harness import load_oracle
from tests.qualification.evol_x.metrics.m03_causal_audit import (
    build_m03_causal_audit,
    verify_causal_audit_against_oracle,
)

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_m03_fault_injection_refuses_bypass(evol_inventory, evol_head) -> None:  # noqa: ANN001
    changed = frozenset(load_oracle("s6_negative_regression")["changed_paths"])
    impact = assess_path_change_impact(evol_inventory, changed_paths=changed)
    base = build_selection_inputs(impact, expected_head_sha=evol_head)
    for row in FAULT_INJECTION_MATRIX:
        decision = select_tests(inject_selection_fault(row.fault, base))
        selected = {n.node_id for n in decision.selected} if decision.selected else set()
        assert decision.status is SelectionStatus.REFUSED_INTEGRITY or (
            decision.status is SelectionStatus.CONSERVATIVE_FULL and base.mandatory_node_ids <= selected
        ), row.fault


def test_m03_causal_audit_passes_on_head(evol_inventory, evol_head) -> None:  # noqa: ANN001
    audit = build_m03_causal_audit(evol_inventory, evol_head)
    assert audit["passed"], audit["rows"]
    oracle = load_evol_x_data_document("m03_oracle_declared.v1.json")
    assert verify_causal_audit_against_oracle(audit, oracle) == []


def test_m03_causal_audit_tampered_oracle_expectation_fails(
    evol_inventory, evol_head, monkeypatch: pytest.MonkeyPatch
) -> None:  # noqa: ANN001
    oracle = copy.deepcopy(load_evol_x_data_document("m03_oracle_declared.v1.json"))
    oracle["arcs"][0]["expected_consumers"] = ["tests/qualification/evol_x/fake.py::test_oracle_tamper"]
    real_loader = load_evol_x_data_document

    def _patched(name: str):  # noqa: ANN001
        if name == "m03_oracle_declared.v1.json":
            return oracle
        return real_loader(name)

    monkeypatch.setattr(
        "tests.qualification.evol_x.metrics.m03_causal_audit.load_evol_x_data_document",
        _patched,
    )
    audit = build_m03_causal_audit(evol_inventory, evol_head)
    assert not audit["passed"]
