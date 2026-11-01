# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.intake.git_change_set import rev_parse_head

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

FRZ_EVOL_IDS = tuple(f"FRZ-EVOL-{i:02d}" for i in range(1, 11))


def test_evol_x_frz_rows_have_qualification_hooks() -> None:
    head = rev_parse_head()
    inventory = build_closed_world_inventory(source_commit_sha=head)
    assert inventory.inventory_digest
    assert FRZ_EVOL_IDS == tuple(f"FRZ-EVOL-{i:02d}" for i in range(1, 11))
