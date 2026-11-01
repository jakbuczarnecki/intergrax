# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.discovery.closed_world_gates import run_closed_world_gates
from testing_support.change_safety.intake.git_change_set import rev_parse_head

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


@pytest.mark.parametrize("gate_id", ("CW-01", "CW-02"))
def test_evol_x_closed_world_gate_passes_on_head(gate_id: str) -> None:
    head = rev_parse_head()
    inventory = build_closed_world_inventory(source_commit_sha=head)
    report = run_closed_world_gates(
        inventory,
        owner_rules=load_evol_x_data_document("owner_rules.v1.json"),
        exclusion_register=load_evol_x_data_document("exclusion_register.v1.json"),
    )
    check = report.check(gate_id)
    assert check is not None
    assert check.passed, check.detail


def test_evol_x_closed_world_cw04_tracks_open_exclusion_proofs() -> None:
    head = rev_parse_head()
    inventory = build_closed_world_inventory(source_commit_sha=head)
    report = run_closed_world_gates(
        inventory,
        owner_rules=load_evol_x_data_document("owner_rules.v1.json"),
        exclusion_register=load_evol_x_data_document("exclusion_register.v1.json"),
    )
    cw04 = report.check("CW-04")
    assert cw04 is not None
    assert cw04.passed is False
    assert cw04.metrics["open_exclusions"] == 3
