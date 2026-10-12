# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R3 current-HEAD gate — run once after R3 evidence is stable."""

from __future__ import annotations

import pytest

from tests.qualification.tenant_x._r3_certification import build_r3_qualification_report, summarize_r3_metrics

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_r3_head_q01_qualification_closure_on_current_head() -> None:
    report = build_r3_qualification_report()
    metrics = summarize_r3_metrics(report)
    assert metrics["recovery_families_total"] == 10
    assert report.checkpoint_to_recovery_missing == 0
    assert report.checkpoint_to_recovery_invalid == 0
    assert report.critical_unknown_families == 0
    assert report.cross_tenant_allowed == 0
    assert report.cross_tenant_unknown == 0
    assert report.authority_mint == 0
    assert report.authority_unknown == 0
    assert report.substitution_illegal_present == 0
    assert report.substitution_unknown == 0
    assert report.durable_key_missing == 0
    assert report.durable_key_invalid == 0
    assert report.r3_ready_for_audit
