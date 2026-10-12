# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R2 fast gates — provider/config/credential isolation (no full P0 rescan)."""

from __future__ import annotations

import subprocess

import pytest

from tests.qualification.tenant_x._r2_adversarial import (
    assert_cross_tenant_credential_ref_rejects,
    assert_missing_tenant_on_credential_scope_rejects,
    assert_reference_executor_rejects_evidence_tenant_mismatch,
    assert_vector_scope_cross_tenant_rejects,
)
from tests.qualification.tenant_x._r2_certification import build_r2_qualification_report, summarize_r2_metrics
from tests.qualification.tenant_x._r2_chain import build_r2_e2e_chain
from tests.qualification.tenant_x._r2_matrices import (
    CREDENTIAL_BROKER_MATRIX,
    CREDENTIAL_REFERENCE_MATRIX,
    CROSS_TENANT_CREDENTIAL_MATRIX,
    MISSING_DEFAULT_GLOBAL_MATRIX,
    PROVIDER_CONFIG_CONTINUITY_MATRIX,
    PROVIDER_CONSUMPTION_MATRIX,
    PROVIDER_PLUGIN_REWRITE_MATRIX,
)
from tests.qualification.tenant_x._r2_mechanisms import build_r2_canonical_mechanisms
from tests.qualification.tenant_x._r2_types import R2ChainVerdict, R2CrossTenantClass, R2SubstitutionClass

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_r2_q01_head_is_git_object() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert len(head) == 40


def test_tx_r2_q02_canonical_mechanism_inventory_bounded() -> None:
    mechanisms = build_r2_canonical_mechanisms()
    assert len(mechanisms) == 9


def test_tx_r2_q03_e2e_chain_transitions_present() -> None:
    chain = build_r2_e2e_chain()
    assert len(chain) == 6
    assert all(t.fail_closed for t in chain)
    assert all(t.verdict == R2ChainVerdict.PROVEN_CURRENT_HEAD for t in chain)


def test_tx_r2_q04_required_matrices_non_empty() -> None:
    assert len(PROVIDER_CONFIG_CONTINUITY_MATRIX) == 9
    assert len(CREDENTIAL_REFERENCE_MATRIX) >= 3
    assert len(CREDENTIAL_BROKER_MATRIX) >= 4
    assert len(PROVIDER_CONSUMPTION_MATRIX) == 9
    assert len(CROSS_TENANT_CREDENTIAL_MATRIX) >= 7
    assert len(PROVIDER_PLUGIN_REWRITE_MATRIX) >= 4
    assert len(MISSING_DEFAULT_GLOBAL_MATRIX) >= 4


def test_tx_r2_q05_adversarial_cross_tenant_credential_broker() -> None:
    assert_cross_tenant_credential_ref_rejects()


def test_tx_r2_q06_adversarial_missing_tenant_credential_scope() -> None:
    assert_missing_tenant_on_credential_scope_rejects()


def test_tx_r2_q07_adversarial_vector_scope_cross_tenant() -> None:
    assert_vector_scope_cross_tenant_rejects()


def test_tx_r2_q08_adversarial_reference_executor_evidence_tenant() -> None:
    assert_reference_executor_rejects_evidence_tenant_mismatch()


def test_tx_r2_q09_closure_derived_security_metrics() -> None:
    report = build_r2_qualification_report()
    assert report.consumption_missing == 0
    assert report.consumption_invalid == 0
    assert report.cross_tenant_allowed == 0
    assert report.cross_tenant_unknown == 0
    assert report.cross_tenant_rejected > 0
    assert report.provider_rewrite_allowed == 0
    assert report.provider_rewrite_unknown == 0
    assert report.substitution_illegal_present == 0
    assert report.substitution_unknown == 0
    assert report.substitution_illegal_input_rejected == 3
    rejected_rows = sum(
        1 for r in MISSING_DEFAULT_GLOBAL_MATRIX if r.verdict == R2SubstitutionClass.ILLEGAL_INPUT_REJECTED
    )
    assert report.substitution_illegal_input_rejected == rejected_rows
    assert report.cross_tenant_rejected == sum(
        1 for r in CROSS_TENANT_CREDENTIAL_MATRIX if r.verdict == R2CrossTenantClass.CROSS_TENANT_REJECTED
    )


def test_tx_r2_q10_report_ready_for_audit() -> None:
    report = build_r2_qualification_report()
    metrics = summarize_r2_metrics(report)
    assert metrics["canonical_mechanisms_total"] == 9
    assert report.r2_ready_for_audit
    assert not report.remediation_children


def test_tx_r2_q11_b2b_nordea_capped_verdict() -> None:
    report = build_r2_qualification_report()
    assert report.b2b_nordea_verdict == "READY WITH PLATFORM GAPS"
