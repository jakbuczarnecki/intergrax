# © Artur Czarnecki. All rights reserved.

"""TENANT-X-P0 fast gates — closed-world inventory and synthetic adversarial probes."""

from __future__ import annotations

import subprocess

import pytest

from tests.qualification.tenant_x._coverage_matrix import TENANT_PROOF_COVERAGE_MATRIX
from tests.qualification.tenant_x._discovery import (
    build_closed_world_discovery_report,
    discover_all_candidates,
)
from tests.qualification.tenant_x._evidence_reconciliation import (
    HISTORICAL_TENANT_EVIDENCE,
    evidence_items_requiring_current_head_replay,
    p0_baseline_matches_head,
)
from tests.qualification.tenant_x._inventory import (
    TENANT_X_INVENTORY,
    inventory_surface_ids,
    unknown_propagation_role_count,
)
from tests.qualification.tenant_x._owner_graph import TENANT_OWNER_GRAPH
from tests.qualification.tenant_x._p0_matrix import build_p0_matrix
from tests.qualification.tenant_x._propagation import TENANT_PROPAGATION_EDGES
from tests.qualification.tenant_x._synthetic import (
    SYNTHETIC_ADVERSARIAL_SCENARIOS,
    evaluate_synthetic_probe,
)
from tests.qualification.tenant_x._types import (
    TENANT_X_P0_BASELINE_SHA,
    TenantBlockerClass,
    TenantIdentitySource,
    TenantPropagationRole,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_p0_q01_baseline_sha() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert p0_baseline_matches_head(head)
    assert head == TENANT_X_P0_BASELINE_SHA


def test_tx_p0_q02_inventory_covers_all_candidates() -> None:
    report = build_closed_world_discovery_report()
    discovered = frozenset(c.surface_id for c in report.raw_candidates)
    classified = inventory_surface_ids()
    assert discovered == classified
    assert report.unclassified_candidate_ids == frozenset()


def test_tx_p0_q03_unknown_propagation_roles_zero() -> None:
    assert unknown_propagation_role_count() == 0
    assert all(r.propagation_role != TenantPropagationRole.UNKNOWN for r in TENANT_X_INVENTORY)
    assert all(r.identity_source != TenantIdentitySource.UNKNOWN for r in TENANT_X_INVENTORY)


def test_tx_p0_q04_canonical_identity_inventory_nonzero() -> None:
    matrix = build_p0_matrix()
    assert matrix.production_files_scanned > 0
    assert matrix.tenant_related_candidates > 0
    assert matrix.canonical_identity_sources >= 9
    assert matrix.unknown_identity_sources == 0
    assert matrix.unknown_propagation_roles == 0


def test_tx_p0_q05_propagation_edges_reference_known_sources() -> None:
    known = inventory_surface_ids()
    for edge in TENANT_PROPAGATION_EDGES:
        assert edge.source_surface_id in known
        assert edge.target_surface_id in known
        assert edge.expected_relation.value in {"SAME", "NARROWER", "EXPLICIT_CROSS_TENANT"}


def test_tx_p0_q06_owner_graph_unique_authority() -> None:
    owners = [item.owner for item in TENANT_OWNER_GRAPH]
    responsibilities = [item.responsibility for item in TENANT_OWNER_GRAPH]
    assert len(responsibilities) == len(set(responsibilities))
    assert "TENANT-X qualification" not in " ".join(owners)


@pytest.mark.parametrize("scenario", SYNTHETIC_ADVERSARIAL_SCENARIOS, ids=lambda s: s.probe_id.value)
def test_tx_p0_q07_synthetic_adversarial_probe(scenario) -> None:
    assert evaluate_synthetic_probe(scenario)


def test_tx_p0_q08_frz_ten_01_counts_reported_not_pass() -> None:
    matrix = build_p0_matrix()
    assert matrix.ambient_default_identity_sources >= 1
    assert matrix.canonical_identity_sources >= 1


def test_tx_p0_q09_legal_cross_tenant_production_mechanisms_zero() -> None:
    matrix = build_p0_matrix()
    assert matrix.legal_cross_tenant_boundaries == 0


def test_tx_p0_q10_historical_evidence_requires_replay() -> None:
    assert len(HISTORICAL_TENANT_EVIDENCE) >= 8
    assert all(item.current_head_replay_required for item in evidence_items_requiring_current_head_replay())


def test_tx_p0_q11_proof_coverage_matrix_complete() -> None:
    concerns = {row.concern for row in TENANT_PROOF_COVERAGE_MATRIX}
    required = {
        "execution",
        "state",
        "provider/config",
        "credentials/secrets",
        "evidence/trace",
        "recovery",
        "extensions",
        "missing tenant",
        "cross-tenant denial",
    }
    assert required <= concerns


def test_tx_p0_q12_in_scope_blockers_catalogued() -> None:
    blockers = [r for r in TENANT_X_INVENTORY if r.blocker_class == TenantBlockerClass.IN_SCOPE_BLOCKER]
    assert blockers, "P0 must classify known ambient/default production blockers"
    cred_gap = [r for r in TENANT_PROOF_COVERAGE_MATRIX if r.concern == "credentials/secrets"][0]
    assert cred_gap.missing_proof is True
