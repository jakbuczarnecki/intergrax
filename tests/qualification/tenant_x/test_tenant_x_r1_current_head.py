# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 current-HEAD gate — run once after R1 evidence is stable."""

from __future__ import annotations

import pytest

from tests.qualification.tenant_x._inventory import get_tenant_x_inventory
from tests.qualification.tenant_x._r1_certification import build_r1_qualification_report, summarize_r1_metrics

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_r1_head_q01_inventory_reconciles_twenty_two_mechanisms() -> None:
    inventory = get_tenant_x_inventory()
    report = build_r1_qualification_report(inventory)
    assert report.authority_mechanisms_total == 22
    metrics = summarize_r1_metrics(report)
    assert metrics["authority_mechanisms_total"] == 22
    assert report.proven_current_head_edges + report.historical_replay_edges + report.missing_edges == 7
    assert report.critical_authority_participants == 0
    assert report.unknown_authority_participation == 0
    assert report.duplicate_confirmed == 0
    assert report.duplicate_unresolved == 0
    assert report.r1_ready_for_audit
