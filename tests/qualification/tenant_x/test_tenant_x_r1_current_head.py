# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 current-HEAD gate — run once after R1 evidence is stable."""

from __future__ import annotations

import pytest

from tests.qualification.tenant_x._inventory import get_tenant_x_inventory
from tests.qualification.tenant_x._r1_certification import build_r1_qualification_report, r1_closure_passes
from tests.qualification.tenant_x._r1_authority_resolutions import build_r1_authority_resolutions

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_r1_head_q01_inventory_reconciles_twenty_two_mechanisms() -> None:
    inventory = get_tenant_x_inventory()
    resolutions = build_r1_authority_resolutions(inventory)
    assert len(resolutions) == 22
    report = build_r1_qualification_report(inventory)
    assert report.authority_mechanisms_total == 22
    assert r1_closure_passes(report)
