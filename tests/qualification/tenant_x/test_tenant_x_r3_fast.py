# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R3 fast gates — checkpoint/recovery tenant continuity (no full STATE-X replay)."""

from __future__ import annotations

import subprocess

import pytest

from tests.qualification.tenant_x._r3_adversarial import (
    assert_agent_checkpoint_cross_tenant_load_rejects,
    assert_checkpoint_recovery_cross_tenant_rejects,
    assert_missing_tenant_resume_rejects_no_global,
    assert_partial_recovery_tenant_mismatch_rejects,
    assert_retry_resume_preserves_tenant_a,
    assert_tenant_b_cannot_load_tenant_a_durable_checkpoint,
)
from tests.qualification.tenant_x._r3_certification import build_r3_qualification_report, summarize_r3_metrics
from tests.qualification.tenant_x._r3_chain import build_r3_e2e_chain
from tests.qualification.tenant_x._r3_matrices import (
    CHECKPOINT_LOAD_MATRIX,
    CHECKPOINT_WRITE_MATRIX,
    CROSS_TENANT_RECOVERY_MATRIX,
    DURABLE_KEY_ISOLATION_MATRIX,
    MISSING_DEFAULT_GLOBAL_MATRIX,
    RECOVERY_RESUME_MATRIX,
    RECOVERY_TENANT_AUTHORITY_MATRIX,
    RETRY_REPLAY_MATRIX,
    STATE_TENANT_CONTINUITY_MATRIX,
)
from tests.qualification.tenant_x._r3_mechanisms import build_r3_recovery_families, r3_critical_unknown_families
from tests.qualification.tenant_x._r3_types import R3ChainVerdict, R3CrossTenantClass, R3SubstitutionClass

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_r3_q01_head_is_git_object() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert len(head) == 40


def test_tx_r3_q02_recovery_family_inventory_bounded() -> None:
    families = build_r3_recovery_families()
    assert len(families) == 10
    assert not r3_critical_unknown_families()


def test_tx_r3_q03_e2e_chain_transitions_present() -> None:
    chain = build_r3_e2e_chain()
    assert len(chain) == 7
    assert all(t.fail_closed for t in chain)
    assert all(t.verdict == R3ChainVerdict.PROVEN_CURRENT_HEAD for t in chain)


def test_tx_r3_q04_required_matrices_non_empty() -> None:
    assert len(STATE_TENANT_CONTINUITY_MATRIX) >= 2
    assert len(CHECKPOINT_WRITE_MATRIX) >= 3
    assert len(CHECKPOINT_LOAD_MATRIX) >= 3
    assert len(RECOVERY_RESUME_MATRIX) >= 3
    assert len(RETRY_REPLAY_MATRIX) >= 3
    assert len(DURABLE_KEY_ISOLATION_MATRIX) >= 4
    assert len(RECOVERY_TENANT_AUTHORITY_MATRIX) >= 4
    assert len(MISSING_DEFAULT_GLOBAL_MATRIX) >= 4
    assert len(CROSS_TENANT_RECOVERY_MATRIX) >= 7


def test_tx_r3_q05_adversarial_checkpoint_recovery_cross_tenant() -> None:
    assert_checkpoint_recovery_cross_tenant_rejects()


def test_tx_r3_q06_adversarial_missing_tenant_no_global_recovery() -> None:
    assert_missing_tenant_resume_rejects_no_global()


def test_tx_r3_q07_adversarial_durable_load_cross_tenant(tmp_path) -> None:
    assert_tenant_b_cannot_load_tenant_a_durable_checkpoint(tmp_path / "task_cp.db")
    assert_agent_checkpoint_cross_tenant_load_rejects()


def test_tx_r3_q08_adversarial_partial_recovery_tenant() -> None:
    assert_partial_recovery_tenant_mismatch_rejects()


def test_tx_r3_q09_retry_resume_preserves_tenant() -> None:
    assert_retry_resume_preserves_tenant_a()


def test_tx_r3_q10_closure_derived_security_metrics() -> None:
    report = build_r3_qualification_report()
    assert report.checkpoint_to_recovery_missing == 0
    assert report.checkpoint_to_recovery_invalid == 0
    assert report.cross_tenant_allowed == 0
    assert report.cross_tenant_unknown == 0
    assert report.cross_tenant_rejected > 0
    assert report.authority_mint == 0
    assert report.authority_unknown == 0
    assert report.substitution_illegal_present == 0
    assert report.substitution_unknown == 0
    assert report.cross_tenant_rejected == sum(
        1 for r in CROSS_TENANT_RECOVERY_MATRIX if r.verdict == R3CrossTenantClass.CROSS_TENANT_REJECTED
    )


def test_tx_r3_q11_report_ready_for_audit() -> None:
    report = build_r3_qualification_report()
    metrics = summarize_r3_metrics(report)
    assert metrics["recovery_families_total"] == 10
    assert report.r3_ready_for_audit
    assert report.remediation_children == ()
