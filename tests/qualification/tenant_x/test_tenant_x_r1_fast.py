# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 fast gates — classifiers, adversarial probes, closure (no full P0 rescan)."""

from __future__ import annotations

import subprocess

import pytest

from tests.qualification.tenant_x._r1_adversarial import (
    assert_functional_evidence_rejects_missing_tenant,
    assert_root_admission_missing_tenant_fails_closed,
    assert_runtime_request_tenant_a_cannot_become_b,
)
from tests.qualification.tenant_x._r1_authority_resolutions import (
    SANCTIONED_ROOT_TENANT_SELECTORS,
    build_r1_authority_resolutions,
    build_r1_overlay_mechanisms,
)
from tests.qualification.tenant_x._r1_certification import (
    LEGAL_CROSS_TENANT_PRODUCTION_MECHANISMS,
    build_r1_qualification_report,
    r1_closure_passes,
    summarize_r1_metrics,
)
from tests.qualification.tenant_x._r1_critical_boundaries import (
    R1_CRITICAL_BOUNDARY_FAMILIES,
    r1_critical_boundary_missing_or_invalid,
)
from tests.qualification.tenant_x._r1_propagation_proofs import (
    certify_r1_propagation_edges,
    r1_critical_propagation_missing,
)
from tests.qualification.tenant_x._r1_silent_substitution import (
    r1_critical_unknown_substitutions,
    r1_illegal_critical_substitutions,
)
from tests.qualification.tenant_x._r1_types import R1ProvenanceStatus
from tests.qualification.tenant_x._types import TenantAuthorityOwnerDisposition, TenantAuthorityScopeEvidence

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_r1_q01_head_is_git_object() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert len(head) == 40


def test_tx_r1_q02_critical_boundary_families_a_through_j() -> None:
    ids = [f.family_id for f in R1_CRITICAL_BOUNDARY_FAMILIES]
    assert ids == ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def test_tx_r1_q03_seven_propagation_edges_proven_current_head() -> None:
    edges = certify_r1_propagation_edges()
    assert len(edges) == 7
    assert r1_critical_propagation_missing() == ()
    assert all(e.final_status == R1ProvenanceStatus.PROVEN_CURRENT_HEAD for e in edges)


def test_tx_r1_q04_twenty_two_authority_mechanisms_resolved() -> None:
    resolutions = build_r1_authority_resolutions()
    assert len(resolutions) == 22
    assert all(
        r.scope_evidence
        not in {TenantAuthorityScopeEvidence.DOMAIN_FALLBACK, TenantAuthorityScopeEvidence.UNKNOWN}
        for r in resolutions
    )
    assert all(r.owner_disposition == TenantAuthorityOwnerDisposition.OWNER_CONFIRMED for r in resolutions)


def test_tx_r1_q05_exactly_one_sanctioned_root_tenant_selector() -> None:
    assert len(SANCTIONED_ROOT_TENANT_SELECTORS) == 1
    overlay = build_r1_overlay_mechanisms()
    roots = [m for m in overlay if m.decision_kind.value == "ROOT_TENANT_SELECTION"]
    assert roots
    non_competing = (
        "application",
        "conformance",
        "cli",
        "task_envelope",
    )
    assert all(any(token in m.authority_scope for token in non_competing) for m in roots)


def test_tx_r1_q06_legal_cross_tenant_production_zero() -> None:
    assert LEGAL_CROSS_TENANT_PRODUCTION_MECHANISMS == ()


def test_tx_r1_q07_silent_substitution_critical_paths_classified() -> None:
    assert r1_critical_unknown_substitutions() == ()
    assert r1_illegal_critical_substitutions() == ()


def test_tx_r1_q08_adversarial_runtime_request_tenant_mismatch() -> None:
    assert_runtime_request_tenant_a_cannot_become_b()


def test_tx_r1_q09_adversarial_root_admission_missing_tenant() -> None:
    assert_root_admission_missing_tenant_fails_closed()


def test_tx_r1_q10_adversarial_functional_evidence_missing_tenant() -> None:
    assert_functional_evidence_rejects_missing_tenant()


def test_tx_r1_q11_r1_closure_gate() -> None:
    report = build_r1_qualification_report()
    metrics = summarize_r1_metrics(report)
    assert metrics["authority_mechanisms_total"] == 22
    assert r1_critical_boundary_missing_or_invalid() == ()
    assert r1_closure_passes(report)


def test_tx_r1_q12_b2b_nordea_capped_verdict() -> None:
    report = build_r1_qualification_report()
    assert report.b2b_nordea_verdict == "READY WITH PLATFORM GAPS"
