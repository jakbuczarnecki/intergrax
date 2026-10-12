# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0-R1 negative integrity / mutation gates (D1–D9)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.qualification.prod_q._p0_certification import (
    build_provider_dispositions,
    build_r1_sqlite_disposition,
    build_tenant_isolation_audit,
)
from tests.qualification.prod_q._p0_classification import classify_candidate, classification_by_candidate_id
from tests.qualification.prod_q._p0_discovery import (
    _module_defines_symbol,
    raw_discovery_candidates,
)
from tests.qualification.prod_q._p0_frz_reconciliation import build_frz_reconciliation
from tests.qualification.prod_q._p0_integrity import (
    assert_inventory_identity_sound,
    probe_duplicate_candidate_id_detected,
    probe_fake_semantic_owner_not_resolved,
    probe_mutated_owner_alters_reconciliation,
    probe_removed_classification_detected,
    probe_unresolved_composition_marks_incomplete,
    probe_unresolved_contract_marks_incomplete,
    probe_unknown_indicator_classified_not_pq_f01,
)
from tests.qualification.prod_q._p0_ownership import (
    probe_alternate_contract_conflict_detected,
    probe_concern_identity_stable_under_candidate_order,
    probe_distinct_concerns_not_merged,
)
from tests.qualification.prod_q._p0_types import SqliteReproductionClassification
from tests.qualification.prod_q._p0_types import (
    CandidateInventoryStatus,
    DiscoveryKind,
    ProductionReadinessCandidate,
    ProductionReadinessEvidenceState,
    ProductionReadinessFamily,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_prod_q_p0_int_d1_unknown_indicator_blocks() -> None:
    assert probe_unknown_indicator_classified_not_pq_f01() is None


def test_prod_q_p0_int_d2_removed_classification_fails_identity() -> None:
    by_id = classification_by_candidate_id()
    assert probe_removed_classification_detected(by_id) is None


def test_prod_q_p0_int_d2_identity_not_index_tautology() -> None:
    raw = tuple(reversed(raw_discovery_candidates()))
    by_id = classification_by_candidate_id()
    assert_inventory_identity_sound(raw, by_id)


def test_prod_q_p0_int_d2_duplicate_candidate_id_mutation() -> None:
    assert probe_duplicate_candidate_id_detected() is None


def test_prod_q_p0_int_d3_fake_semantic_owner_not_resolved() -> None:
    assert probe_fake_semantic_owner_not_resolved() is None


def test_prod_q_p0_int_d5_tenant_not_static_pass_with_contradiction() -> None:
    audit = build_tenant_isolation_audit()
    assert audit.verdict != "PASS"
    mutant = replace(audit, verdict="PASS", production_path_contradictions=("forced",))
    assert mutant.production_path_contradictions != ()


def test_prod_q_p0_int_d6_frz_no_proven_without_gate() -> None:
    for record in build_frz_reconciliation():
        if record.evidence_state == ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD:
            assert "gate:" in " ".join(record.evidence_refs)


def test_prod_q_p0_int_d6_frz_registry_33_unique() -> None:
    records = build_frz_reconciliation()
    ids = [r.criterion_id for r in records]
    assert len(ids) == 33
    assert len(frozenset(ids)) == 33


def test_prod_q_p0_int_d7_sqlite_reproduction_executed() -> None:
    record = build_r1_sqlite_disposition()
    assert record.reproduction_executed is True
    assert record.reproduction_exit_code is not None
    assert record.reproduction_classification != SqliteReproductionClassification.PRODUCTION_DEFECT
    assert record.strict_dependency_materialization_observed is True
    assert "ToolDependencyAttemptBoundaryMaterializationError" in record.failing_exception_type


def test_prod_q_p0_int_d8_provider_bundle_not_registered_alone() -> None:
    for provider in build_provider_dispositions():
        if provider.implementation_exists and not provider.registration_evidence:
            assert provider.registered is False


def test_prod_q_p0_int_d9_missing_anchor_symbol_fails_resolution() -> None:
    ok, _line, reason = _module_defines_symbol(
        "intergrax/hosting/engine/engine.py",
        "__prod_q_nonexistent_symbol__",
    )
    assert ok is False
    assert reason


def test_prod_q_p0_int_d9_method_level_health_discovered() -> None:
    method_candidates = [
        c
        for c in raw_discovery_candidates()
        if c.discovery_kind == DiscoveryKind.AST_METHOD_INDICATOR and c.indicator == "health"
    ]
    assert method_candidates, "expected method-level health mechanism in closed-world discovery"


def test_prod_q_p0_int_d1_gate_changes_on_unknown_indicator() -> None:
    baseline_unclassified = sum(
        1
        for r in classification_by_candidate_id().values()
        if r.inventory_status == CandidateInventoryStatus.UNCLASSIFIED_INDICATOR
    )
    assert baseline_unclassified == 0
    sample = next(iter(raw_discovery_candidates()))
    mutant = classify_candidate(replace(sample, indicator="__unknown__"))
    assert mutant.inventory_status == CandidateInventoryStatus.UNCLASSIFIED_INDICATOR
    assert mutant.family == ProductionReadinessFamily.PQ_UNCLASSIFIED


def test_prod_q_p0_int_ownership_unresolved_contract_incomplete() -> None:
    assert probe_unresolved_contract_marks_incomplete() is None


def test_prod_q_p0_int_ownership_unresolved_composition_incomplete() -> None:
    assert probe_unresolved_composition_marks_incomplete() is None


def test_prod_q_p0_int_ownership_alternate_contract_conflict() -> None:
    assert probe_alternate_contract_conflict_detected() is None


def test_prod_q_p0_int_ownership_distinct_concerns_not_merged() -> None:
    assert probe_distinct_concerns_not_merged() is None


def test_prod_q_p0_int_ownership_concern_identity_order_stable() -> None:
    assert probe_concern_identity_stable_under_candidate_order() is None


def test_prod_q_p0_int_ownership_mutated_state_alters_reconciliation() -> None:
    assert probe_mutated_owner_alters_reconciliation() is None


def test_prod_q_p0_int_d8_nonexistent_provider_not_registered() -> None:
    ghost = ProductionReadinessCandidate(
        candidate_id="ghost::bundle::provider.sandbox_host",
        module_path="intergrax/integrations/providers/sandbox_host/ghost/bundle.py",
        symbol="sandbox_host_bundle",
        discovery_kind=DiscoveryKind.PROVIDER_SANDBOX_HOST,
        line_no=1,
        indicator="sandbox_provider",
    )
    record = classify_candidate(ghost)
    assert record.evidence_state == ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED
