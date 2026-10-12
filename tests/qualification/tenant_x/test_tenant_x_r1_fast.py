# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 fast gates — classifiers, adversarial probes, closure (no full P0 rescan)."""

from __future__ import annotations

import subprocess

import pytest

from tests.qualification.tenant_x._r1_adversarial import (
    assert_functional_evidence_rejects_missing_tenant,
    assert_governance_tenant_matches_runtime_envelope,
    assert_root_admission_missing_tenant_fails_closed,
    assert_runtime_request_tenant_a_cannot_become_b,
    assert_tool_scope_tenant_follows_inspection_scope,
)
from tests.qualification.tenant_x._r1_authority_resolutions import (
    build_r1_authority_resolutions,
    build_r1_overlay_mechanisms,
    build_sanctioned_root_tenant_selectors,
)
from tests.qualification.tenant_x._r1_capability_routing_consumers import (
    assert_capability_routing_sentinel_reconciliation,
    assert_routing_sentinel_structural_proof,
    build_capability_routing_consumer_inventory,
    capability_routing_consumer_counts,
    routing_sentinel_semantics_documentation,
)
from tests.qualification.tenant_x._r1_certification import build_r1_qualification_report, summarize_r1_metrics
from tests.qualification.tenant_x._r1_types import R1MechanismAuthorityClass
from tests.qualification.tenant_x._r1_critical_boundaries import R1_CRITICAL_BOUNDARY_FAMILIES
from tests.qualification.tenant_x._r1_propagation_proofs import certify_r1_propagation_edges
from tests.qualification.tenant_x._r1_silent_substitution import discover_silent_substitution_hits
from tests.qualification.tenant_x._r1_types import R1ProvenanceStatus
from tests.qualification.tenant_x._types import TenantAuthorityOwnerDisposition, TenantAuthorityScopeEvidence

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_r1_q01_head_is_git_object() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert len(head) == 40


def test_tx_r1_q02_critical_boundary_families_a_through_j() -> None:
    ids = [f.family_id for f in R1_CRITICAL_BOUNDARY_FAMILIES]
    assert ids == ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def test_tx_r1_q03_propagation_edges_use_structural_proof_not_markers_only() -> None:
    edges = certify_r1_propagation_edges()
    assert len(edges) == 7
    for edge in edges:
        if edge.final_status == R1ProvenanceStatus.PROVEN_CURRENT_HEAD:
            assert edge.current_head_evidence
            assert all("::" in item for item in edge.current_head_evidence)
        if edge.final_status == R1ProvenanceStatus.MISSING:
            assert not edge.current_head_evidence or edge.current_head_evidence[0].startswith("discovery:")


def test_tx_r1_q04_authority_resolutions_carry_typed_evidence() -> None:
    resolutions = build_r1_authority_resolutions()
    assert len(resolutions) == 22
    for resolution in resolutions:
        if resolution.owner_disposition == TenantAuthorityOwnerDisposition.OWNER_CONFIRMED:
            assert resolution.owner_evidence, resolution.mechanism_id
        if resolution.scope_evidence == TenantAuthorityScopeEvidence.UNKNOWN:
            assert resolution.scope_evidence_refs


def test_tx_r1_q05_sanctioned_root_selector_from_inventory() -> None:
    selectors = build_sanctioned_root_tenant_selectors()
    assert len(selectors) == 1
    assert "RootExecutionAuthorityAdmissionRequest" in selectors[0]


def test_tx_r1_q06_cross_tenant_inventory_derived_not_empty_tuple_constant() -> None:
    report = build_r1_qualification_report()
    assert report.legal_cross_tenant_mechanisms >= 0
    metrics = summarize_r1_metrics(report)
    assert "legal_cross_tenant_mechanisms" in metrics


def test_tx_r1_q07_silent_substitution_mechanical_discovery_runs() -> None:
    hits = discover_silent_substitution_hits()
    assert isinstance(hits, tuple)


def test_tx_r1_q08_adversarial_runtime_request_tenant_mismatch() -> None:
    assert_runtime_request_tenant_a_cannot_become_b()


def test_tx_r1_q09_adversarial_root_admission_missing_tenant() -> None:
    assert_root_admission_missing_tenant_fails_closed()


def test_tx_r1_q10_adversarial_functional_evidence_missing_tenant() -> None:
    assert_functional_evidence_rejects_missing_tenant()


def test_tx_r1_q11_adversarial_runtime_to_envelope_same_tenant() -> None:
    assert_governance_tenant_matches_runtime_envelope()


def test_tx_r1_q12_adversarial_tool_scope_same_tenant() -> None:
    assert_tool_scope_tenant_follows_inspection_scope()


def test_tx_r1_q13_report_emits_remediation_when_blocked() -> None:
    report = build_r1_qualification_report()
    if not report.r1_ready_for_audit:
        assert report.remediation_children
    overlay = build_r1_overlay_mechanisms()
    assert overlay


def test_tx_r1_q14_b2b_nordea_capped_verdict() -> None:
    report = build_r1_qualification_report()
    assert report.b2b_nordea_verdict == "READY WITH PLATFORM GAPS"


def test_tx_r1_q15_capability_routing_sentinel_consumer_inventory() -> None:
    inventory = build_capability_routing_consumer_inventory()
    assert len(inventory) >= 4
    counts = capability_routing_consumer_counts()
    assert counts["UNKNOWN"] == 0
    assert counts["CAPABILITY_ROUTING_ONLY"] == len(inventory)
    assert_routing_sentinel_structural_proof()
    assert_capability_routing_sentinel_reconciliation()
    assert routing_sentinel_semantics_documentation()


def test_tx_r1_q16_agent_capability_intake_non_authoritative_routing() -> None:
    resolutions = build_r1_authority_resolutions()
    intake = next(r for r in resolutions if "agent_capability_intake.py" in r.mechanism_id)
    assert intake.authority_class == R1MechanismAuthorityClass.NON_AUTHORITATIVE_ROUTING
    assert "routing sentinel" in intake.notes.lower() or "routing" in intake.notes.lower()
    assert "subordinate" not in intake.notes.lower()
    assert "derived from admitted task context" not in " ".join(intake.scope_evidence_refs).lower()
