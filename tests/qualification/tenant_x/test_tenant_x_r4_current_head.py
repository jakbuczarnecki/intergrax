# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R4 current-HEAD gate — run once after R4 evidence is stable."""

from __future__ import annotations

import pytest

from tests.qualification.tenant_x._r4_certification import build_r4_qualification_report, summarize_r4_metrics

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_r4_head_q01_qualification_closure_on_current_head() -> None:
    report = build_r4_qualification_report()
    metrics = summarize_r4_metrics(report)
    assert metrics["r4_families_total"] == 10
    assert report.context_model_gap_proven
    assert report.context_model_missing == 0
    assert report.context_model_invalid == 0
    assert report.critical_unknown_families == 0
    assert report.cross_tenant_allowed == 0
    assert report.cross_tenant_unknown == 0
    assert report.observability_authority == 0
    assert report.observability_unknown == 0
    assert report.extension_allowed == 0
    assert report.extension_unknown == 0
    assert report.substitution_illegal_present == 0
    assert report.substitution_unknown == 0
    assert report.queue_worker_missing == 0
    assert report.queue_worker_invalid == 0
    assert report.async_worker_missing == 0
    assert report.async_worker_invalid == 0
    assert report.evidence_missing == 0
    assert report.evidence_invalid == 0
    assert report.r4_ready_for_audit
