# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.assess.workflow_evolution import (
    WorkflowChange,
    classify_inflight_definition_change,
    classify_workflow_change,
    enumerate_input_space,
    load_workflow_matrix,
    matrix_row_coverage,
    unsafe_inflight_adoptions,
)
from testing_support.change_safety.contracts.enums import WorkflowEvolutionDecision as D

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_matrix_has_wm_01_to_15_and_fail_closed_tail() -> None:
    rows = load_workflow_matrix()
    assert {r["wm_id"] for r in rows} == {f"WM-{i:02d}" for i in range(1, 16)}
    assert rows[-1]["wm_id"] == "WM-15" and rows[-1]["decision"] == "REJECT"


def test_every_matrix_row_is_reachable_and_input_space_is_total() -> None:
    assert len(enumerate_input_space()) == 128
    coverage = matrix_row_coverage()
    unreachable = sorted(k for k, v in coverage.items() if v == 0)
    # WM-15 is the fail-closed catch-all: it may legitimately be unreachable for a total matrix.
    assert unreachable in ([], ["WM-15"]), unreachable
    assert sum(coverage.values()) == 128


def test_inflight_in_place_adoption_is_never_permissive() -> None:
    assert unsafe_inflight_adoptions() == ()


@pytest.mark.parametrize(
    ("change", "matrix_id", "decision"),
    [
        (WorkflowChange(has_inflight=False, fingerprint_changed=False), "WM-01", D.SAFE_IN_PLACE),
        (WorkflowChange(has_inflight=True, adopt_in_place=True, pin_evidence=False), "WM-05", D.EXPLICITLY_UNSUPPORTED),
        (
            WorkflowChange(has_inflight=True, adopt_in_place=True, graph_spec_bumped=True),
            "WM-06",
            D.EXPLICITLY_UNSUPPORTED,
        ),
        (WorkflowChange(has_inflight=True, adopt_in_place=True), "WM-07", D.EXPLICITLY_UNSUPPORTED),
        (WorkflowChange(has_inflight=True, adopt_in_place=False), "WM-09", D.VERSIONED_DEPLOY),
        (WorkflowChange(has_inflight=True, adopt_in_place=False, pin_evidence=False), "WM-10", D.REJECT),
        (WorkflowChange(has_inflight=False, migration_declared=True), "WM-11", D.MIGRATE),
        (WorkflowChange(has_inflight=False, superseded_readable=False), "WM-02", D.REJECT),
        (WorkflowChange(has_inflight=False, graph_spec_bumped=True), "WM-04", D.REJECT),
        (
            WorkflowChange(has_inflight=False, graph_spec_bumped=True, migration_declared=True),
            "WM-03",
            D.MIGRATE,
        ),
    ],
)
def test_representative_rows(change: WorkflowChange, matrix_id: str, decision: D) -> None:
    verdict = classify_workflow_change(change)
    assert (verdict.matrix_id, verdict.decision) == (matrix_id, decision)
    assert verdict.witness


def test_backward_compatible_entry_point_matches_matrix() -> None:
    assert classify_inflight_definition_change(has_inflight=True, migration_declared=False) == D.EXPLICITLY_UNSUPPORTED
    assert classify_inflight_definition_change(has_inflight=False, migration_declared=True) == D.MIGRATE


def test_negative_matrix_tampering_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.change_safety.assess import workflow_evolution as mod

    real = mod.load_evol_x_data_document("workflow_evolution_matrix.v1.json")
    broken = dict(real, rows=[r for r in real["rows"] if r["wm_id"] != "WM-15"])
    monkeypatch.setattr(mod, "load_evol_x_data_document", lambda _name: broken)
    with pytest.raises(ValueError, match="catch-all"):
        mod.load_workflow_matrix()
