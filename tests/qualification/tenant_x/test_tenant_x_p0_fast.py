# © Artur Czarnecki. All rights reserved.

"""TENANT-X-P0-R1 fast gates — classifiers, provenance, synthetic probes (no full inventory scan)."""

from __future__ import annotations

import subprocess

import pytest

from tests.qualification.tenant_x._classifiers import (
    classify_source_fixture,
    domain_for_path,
    uses_generic_safe_fallback,
)
from tests.qualification.tenant_x._types import TenantPersistenceRole
from tests.qualification.tenant_x._coverage_matrix import TENANT_PROOF_COVERAGE_MATRIX
from tests.qualification.tenant_x._evidence_reconciliation import (
    HISTORICAL_TENANT_EVIDENCE,
    evidence_items_requiring_current_head_replay,
    historical_evidence_sha_mismatches,
)
from tests.qualification.tenant_x._owner_graph import TENANT_OWNER_GRAPH
from tests.qualification.tenant_x._propagation import (
    TENANT_PROPAGATION_GAPS,
    validate_propagation_edges,
)
from tests.qualification.tenant_x._synthetic import (
    CLASSIFIER_PROBE_SCENARIOS,
    SYNTHETIC_ADVERSARIAL_SCENARIOS,
    evaluate_classifier_probe,
    evaluate_synthetic_probe,
)
from tests.qualification.tenant_x._types import TenantDomain, TenantPropagationRole

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_p0_r1_q01_head_is_git_object_not_pinned_task_sha() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert len(head) == 40
    assert head.isalnum()


def test_tx_p0_r1_q02_ctrl_x_historical_evidence_sha() -> None:
    ctrl = next(item for item in HISTORICAL_TENANT_EVIDENCE if item.stage_id == "CTRL-X")
    assert ctrl.accepted_sha == "318a4d835534ef5c48575f4649551c957dacd8ea"
    assert historical_evidence_sha_mismatches() == ()


def test_tx_p0_r1_q03_domain_specificity_credentials_before_integrations() -> None:
    assert (
        domain_for_path("intergrax/integrations/credentials/broker.py")
        == TenantDomain.CREDENTIALS_SECRETS
    )
    assert (
        domain_for_path("intergrax/integrations/foo/bar.py") == TenantDomain.INTEGRATION_RESOLUTION
    )


def test_tx_p0_r1_q04_recovery_checkpoint_events_specificity() -> None:
    assert domain_for_path("intergrax/runtime/recovery/coordinator.py") == TenantDomain.RECOVERY
    assert domain_for_path("intergrax/runtime/checkpoint/store.py") == TenantDomain.CHECKPOINT
    assert domain_for_path("intergrax/runtime/events/bus.py") == TenantDomain.EVENTS


@pytest.mark.parametrize("scenario", CLASSIFIER_PROBE_SCENARIOS, ids=lambda s: s.probe_id.value)
def test_tx_p0_r1_q05_classifier_synthetic_probes(scenario) -> None:
    assert evaluate_classifier_probe(scenario)


@pytest.mark.parametrize("scenario", SYNTHETIC_ADVERSARIAL_SCENARIOS, ids=lambda s: s.probe_id.value)
def test_tx_p0_r1_q06_legacy_bucket_probes(scenario) -> None:
    assert evaluate_synthetic_probe(scenario)


def test_tx_p0_r1_q07_propagation_gaps_not_surrogate_edges() -> None:
    gap_ids = {g.gap_id for g in TENANT_PROPAGATION_GAPS}
    assert "gap.checkpoint_to_recovery" in gap_ids
    assert "gap.credential_to_provider_consumption" in gap_ids


def test_tx_p0_r1_q08_owner_graph_non_authoritative() -> None:
    owners = [item.owner for item in TENANT_OWNER_GRAPH]
    assert all(o.startswith("CANDIDATE_OWNER:") for o in owners)
    assert "TENANT-X qualification" not in " ".join(owners)


def test_tx_p0_r1_q09_historical_evidence_requires_replay() -> None:
    assert len(HISTORICAL_TENANT_EVIDENCE) >= 8
    assert all(item.current_head_replay_required for item in evidence_items_requiring_current_head_replay())


def test_tx_p0_r1_q10_proof_coverage_matrix_complete() -> None:
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


def test_tx_p0_r1_q11_classifier_no_generic_safe_fallback_on_fixture() -> None:
    for scenario in CLASSIFIER_PROBE_SCENARIOS:
        for record in classify_source_fixture(scenario.module_path, scenario.source):
            assert not uses_generic_safe_fallback(record)


def test_tx_p0_r1_q13_storage_key_index_fixture_semantics() -> None:
    positive = """
def save(item_id: str, tenant_id: str, value: str):
    key = (tenant_id, item_id)
    store[key] = value
"""
    neg = """
class Cache:
    tenant_id: str

    def get(self, item_id: str):
        return store[item_id]
"""
    pos_records = classify_source_fixture("synthetic/probes/key_positive.py", positive)
    neg_records = classify_source_fixture("synthetic/probes/key_negative.py", neg)
    assert any(r.persistence_role == TenantPersistenceRole.TENANT_IN_KEY_OR_PREDICATE for r in pos_records)
    cache = next(r for r in neg_records if r.symbol == "Cache")
    assert cache.persistence_role == TenantPersistenceRole.MISSING_ISOLATION_EVIDENCE


def test_tx_p0_r1_q12_propagation_edge_validation_empty_without_inventory() -> None:
    violations = validate_propagation_edges(frozenset())
    assert violations
