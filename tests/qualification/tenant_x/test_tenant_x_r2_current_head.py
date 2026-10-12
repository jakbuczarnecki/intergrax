# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R2 current-HEAD gate — run once after R2 evidence is stable."""

from __future__ import annotations

import pytest

from tests.qualification.tenant_x._r2_certification import build_r2_qualification_report, summarize_r2_metrics

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_r2_head_q01_qualification_closure_on_current_head() -> None:
    report = build_r2_qualification_report()
    metrics = summarize_r2_metrics(report)
    assert metrics["canonical_mechanisms_total"] == 9
    assert report.consumption_missing == 0
    assert report.consumption_invalid == 0
    assert report.credential_resolution_missing == 0
    assert report.credential_resolution_invalid == 0
    assert report.continuity_missing == 0
    assert report.continuity_invalid == 0
    assert report.cross_tenant_allowed == 0
    assert report.cross_tenant_unknown == 0
    assert report.provider_rewrite_allowed == 0
    assert report.provider_rewrite_unknown == 0
    assert report.substitution_illegal_present == 0
    assert report.substitution_unknown == 0
    assert report.r2_ready_for_audit
