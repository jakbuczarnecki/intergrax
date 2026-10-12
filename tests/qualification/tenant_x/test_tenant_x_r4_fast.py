# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R4 fast gates — context/evidence/async tenant continuity (no full P0 scan)."""

from __future__ import annotations

import subprocess

import pytest

from tests.qualification.tenant_x._r4_adversarial import (
    assert_background_blank_transport_rejects,
    assert_background_transport_task_tenant_mismatch_rejects,
    assert_context_llm_cross_tenant_attribution_rejects,
    assert_context_view_request_authority_mismatch_rejects,
    assert_context_view_request_membership_mismatch_rejects,
    assert_functional_evidence_missing_tenant_rejects,
    assert_vendor_queue_request_tenant_mismatch_rejects,
    assert_vendor_worker_payload_job_tenant_mismatch_rejects,
)
from tests.qualification.tenant_x._r4_certification import build_r4_qualification_report, summarize_r4_metrics
from tests.qualification.tenant_x._r4_chain import build_r4_e2e_chain
from tests.qualification.tenant_x._r4_matrices import (
    ASYNC_DOMAIN_WORKER_MATRIX,
    BACKGROUND_EXECUTION_IDENTITY_MATRIX,
    CONTEXT_MODEL_CONTINUITY_MATRIX,
    CONTEXT_VIEW_ISOLATION_MATRIX,
    CROSS_TENANT_R4_MATRIX,
    EVIDENCE_TENANT_CONTINUITY_MATRIX,
    EVENT_TRACE_CONTINUITY_MATRIX,
    EXTENSION_SCOPE_MATRIX,
    MISSING_DEFAULT_GLOBAL_MATRIX,
    QUEUE_WORKER_CONTINUITY_MATRIX,
)
from tests.qualification.tenant_x._r4_mechanisms import build_r4_families, r4_critical_unknown_families
from tests.qualification.tenant_x._r4_types import R4ChainVerdict, R4CrossTenantClass

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_r4_q01_head_is_git_object() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert len(head) == 40


def test_tx_r4_q02_family_inventory_bounded() -> None:
    families = build_r4_families()
    assert len(families) == 10
    assert not r4_critical_unknown_families()


def test_tx_r4_q03_e2e_chain_transitions_present() -> None:
    chain = build_r4_e2e_chain()
    assert len(chain) == 10
    assert all(t.fail_closed for t in chain)
    assert all(t.verdict == R4ChainVerdict.PROVEN_CURRENT_HEAD for t in chain)


def test_tx_r4_q04_required_matrices_non_empty() -> None:
    assert len(CONTEXT_MODEL_CONTINUITY_MATRIX) >= 3
    assert len(CONTEXT_VIEW_ISOLATION_MATRIX) >= 5
    assert len(EVIDENCE_TENANT_CONTINUITY_MATRIX) >= 2
    assert len(EVENT_TRACE_CONTINUITY_MATRIX) >= 4
    assert len(BACKGROUND_EXECUTION_IDENTITY_MATRIX) >= 4
    assert len(QUEUE_WORKER_CONTINUITY_MATRIX) >= 3
    assert len(ASYNC_DOMAIN_WORKER_MATRIX) >= 3
    assert len(EXTENSION_SCOPE_MATRIX) >= 2
    assert len(MISSING_DEFAULT_GLOBAL_MATRIX) >= 4
    assert len(CROSS_TENANT_R4_MATRIX) >= 7


def test_tx_r4_q05_adversarial_context_llm_cross_tenant() -> None:
    assert_context_llm_cross_tenant_attribution_rejects()


def test_tx_r4_q06_adversarial_context_view_request_mismatch() -> None:
    assert_context_view_request_authority_mismatch_rejects()
    assert_context_view_request_membership_mismatch_rejects()


def test_tx_r4_q07_adversarial_background_identity() -> None:
    assert_background_transport_task_tenant_mismatch_rejects()
    assert_background_blank_transport_rejects()


def test_tx_r4_q08_adversarial_vendor_async_queue() -> None:
    assert_vendor_queue_request_tenant_mismatch_rejects()
    assert_vendor_worker_payload_job_tenant_mismatch_rejects()


def test_tx_r4_q09_adversarial_functional_evidence_tenant() -> None:
    assert_functional_evidence_missing_tenant_rejects()


def test_tx_r4_q10_closure_derived_security_metrics() -> None:
    report = build_r4_qualification_report()
    assert report.context_model_gap_proven
    assert report.context_model_missing == 0
    assert report.context_model_invalid == 0
    assert report.cross_tenant_allowed == 0
    assert report.cross_tenant_unknown == 0
    assert report.observability_authority == 0
    assert report.observability_unknown == 0
    assert report.extension_allowed == 0
    assert report.extension_unknown == 0
    assert report.substitution_illegal_present == 0
    assert report.substitution_unknown == 0
    assert report.cross_tenant_rejected == sum(
        1 for r in CROSS_TENANT_R4_MATRIX if r.verdict == R4CrossTenantClass.CROSS_TENANT_REJECTED
    )


def test_tx_r4_q11_report_ready_for_audit() -> None:
    report = build_r4_qualification_report()
    metrics = summarize_r4_metrics(report)
    assert metrics["r4_families_total"] == 10
    assert report.r4_ready_for_audit
    assert report.remediation_children == ()
