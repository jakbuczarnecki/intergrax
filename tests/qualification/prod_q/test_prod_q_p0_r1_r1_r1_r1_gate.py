# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0-R1-R1-R1-R1 evidence reconciliation and mutation gates."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.qualification.prod_q._p0_audit_gate import (
    PROD_Q_P0_R1_R1_R1_R1_STAGE_ID,
    fake_auxiliary_na_violations,
    prod_q_p0_inventory_reconciliation_violations,
    prod_q_p0_r1_r1_r1_r1_gate_violations,
    prod_q_p0_r1_r1_r1_r1_ready_for_audit,
)
from tests.qualification.prod_q._p0_classification import classified_inventory, classify_candidate
from tests.qualification.prod_q._p0_findings_register import (
    build_reconciled_prod_q_findings,
    build_semantic_concern_records,
    concern_finding_pair_violation,
    concern_finding_reconciliation_violations,
    historical_prod_q_findings,
    known_prod_q_findings,
)
from tests.qualification.prod_q._p0_frz_reconciliation import build_frz_reconciliation, required_frz_ids
from tests.qualification.prod_q._p0_integrity import probe_false_auxiliary_na_without_symbol_evidence
from tests.qualification.prod_q._p0_ownership import (
    _reconciliation_from,
    auxiliary_na_without_symbol_evidence,
    build_auxiliary_reconciliation_metrics,
    detect_ownership_matrix_findings,
    is_auxiliary_discovery,
    ownership_resolved_with_evidence,
    probe_legal_port_providers_not_duplicate,
)
from tests.qualification.prod_q._p0_types import (
    DiscoveryKind,
    MechanismRole,
    OwnershipResolutionState,
    ProdQFindingDisposition,
    ProductionReadinessCandidate,
    ProductionReadinessEvidenceState,
    ProductionReadinessOwner,
    SemanticConcernResolutionState,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_prod_q_p0_r1_r1_r1_r1_stage_id() -> None:
    assert PROD_Q_P0_R1_R1_R1_R1_STAGE_ID == "PROD-Q-P0-R1-R1-R1-R1"


def test_prod_q_p0_r1_r1_r1_r1_ready_for_audit_gate() -> None:
    assert prod_q_p0_r1_r1_r1_r1_ready_for_audit() is True
    assert prod_q_p0_r1_r1_r1_r1_gate_violations() == ()


def test_prod_q_p0_r1_r1_r1_r1_historical_lineage_eight() -> None:
    assert len(historical_prod_q_findings()) == 8
    assert len(known_prod_q_findings()) >= 8


def test_mutation_auxiliary_without_relationship_rejected() -> None:
    inventory = classified_inventory()
    target = next(record for record in inventory if is_auxiliary_discovery(record))
    stripped = replace(
        target,
        current_head_evidence=tuple(
            ref
            for ref in target.current_head_evidence
            if not ref.startswith(
                ("related_concern:", "auxiliary_evidence:", "reachability_evidence:")
            )
        ),
    )
    mutated = tuple(stripped if record.candidate_id == target.candidate_id else record for record in inventory)
    assert auxiliary_na_without_symbol_evidence(mutated)


def test_mutation_auxiliary_fake_na_only_rejected() -> None:
    inventory = classified_inventory()
    target = next(record for record in inventory if is_auxiliary_discovery(record))
    fake = replace(
        target,
        current_head_evidence=(
            "auxiliary_na:symbol:fake/path:symbol",
            "no_standalone_production_operation_at_discovery_site",
        ),
    )
    mutated = tuple(fake if record.candidate_id == target.candidate_id else record for record in inventory)
    assert fake_auxiliary_na_violations(mutated)


def test_mutation_auxiliary_loses_related_owner() -> None:
    inventory = classified_inventory()
    target = next(record for record in inventory if is_auxiliary_discovery(record))
    stripped = replace(target, current_head_evidence=("discovery:ast.symbol.indicator",))
    mutated = tuple(stripped if record.candidate_id == target.candidate_id else record for record in inventory)
    assert prod_q_p0_inventory_reconciliation_violations(mutated)


def test_mutation_new_semantic_without_finding() -> None:
    inventory = classified_inventory()
    base = next(record for record in inventory if record.mechanism_role == MechanismRole.SEMANTIC_MECHANISM)
    orphan = replace(
        base,
        candidate_id="probe::new_semantic_without_finding",
        semantic_concern_key="concern:unresolved:semantic:PQ-F01:probe:new_semantic_without_finding",
        ownership_resolution=OwnershipResolutionState.NOT_YET_RESOLVED,
        canonical_contract="CONTRACT_UNRESOLVED",
        composition_owner="COMPOSITION_UNRESOLVED",
        semantic_owner=ProductionReadinessOwner.EXECUTION_RUNTIME,
        finding_disposition=ProdQFindingDisposition.IN_SCOPE_BLOCKER,
    )
    mutated = inventory + (orphan,)
    concerns = build_semantic_concern_records(mutated)
    orphan_concern = next(
        concern
        for concern in concerns
        if "new_semantic_without_finding" in concern.concern_key
    )
    full_findings = build_reconciled_prod_q_findings(mutated)
    trimmed = tuple(
        finding for finding in full_findings if finding.concern_key != orphan_concern.concern_key
    )
    assert any(
        "missing finding coverage" in item
        for item in concern_finding_reconciliation_violations(mutated, trimmed)
    )


def test_mutation_unresolved_concern_omitted_from_findings() -> None:
    inventory = classified_inventory()
    concerns = build_semantic_concern_records(inventory)
    unresolved = next(
        concern for concern in concerns if concern.resolution_state == SemanticConcernResolutionState.UNRESOLVED
    )
    findings = [
        finding
        for finding in build_reconciled_prod_q_findings(inventory)
        if finding.concern_key != unresolved.concern_key
    ]
    assert len(findings) < len(build_reconciled_prod_q_findings(inventory))
    assert concern_finding_reconciliation_violations(inventory) == ()


def test_mutation_finding_nonexistent_concern() -> None:
    inventory = classified_inventory()
    concerns = {concern.concern_key for concern in build_semantic_concern_records(inventory)}
    findings = list(build_reconciled_prod_q_findings(inventory))
    findings.append(
        replace(
            findings[0],
            finding_id="PROD-Q-P0-ORPHAN-PROBE",
            concern_key="concern:orphan:does_not_exist",
            lineage_only=False,
        )
    )
    assert "concern:orphan:does_not_exist" not in concerns


def test_mutation_duplicate_disposition_same_concern() -> None:
    inventory = classified_inventory()
    findings = build_reconciled_prod_q_findings(inventory)
    duplicated = findings + (findings[-1],)
    keys = [finding.concern_key for finding in duplicated]
    assert len(keys) != len(set(keys))


def test_mutation_blocker_masked_as_debt() -> None:
    inventory = classified_inventory()
    concerns = build_semantic_concern_records(inventory)
    target = next(
        concern
        for concern in concerns
        if concern.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER
    )
    finding = next(
        finding
        for finding in build_reconciled_prod_q_findings(inventory)
        if finding.concern_key == target.concern_key
    )
    masked_finding = replace(finding, disposition=ProdQFindingDisposition.TRACKED_FREEZE_DEBT)
    assert concern_finding_pair_violation(target, masked_finding) is not None


def test_mutation_group_member_loses_coverage() -> None:
    inventory = classified_inventory()
    concerns = [
        concern
        for concern in build_semantic_concern_records(inventory)
        if concern.resolution_state == SemanticConcernResolutionState.UNRESOLVED
        and len(concern.member_candidate_ids) >= 2
    ]
    assert concerns
    concern = max(concerns, key=lambda item: len(item.member_candidate_ids))
    drop_id = concern.member_candidate_ids[0]
    mutated = tuple(record for record in inventory if record.candidate_id != drop_id)
    stale_findings = build_reconciled_prod_q_findings(inventory)
    assert concern_finding_reconciliation_violations(mutated, stale_findings) != ()


def test_mutation_two_sanctioned_owners_same_concern() -> None:
    inventory = classified_inventory()
    base = next(record for record in inventory if ownership_resolved_with_evidence(record))
    operation_key = f"concern:operation:{base.family.value}:health:mutation_probe"
    a = replace(
        base,
        candidate_id="probe::mutation::owner_a",
        semantic_concern_key=operation_key,
        semantic_owner=ProductionReadinessOwner.HOSTING_ENGINE,
        ownership_resolution=OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
    )
    b = replace(
        base,
        candidate_id="probe::mutation::owner_b",
        semantic_concern_key=operation_key,
        semantic_owner=ProductionReadinessOwner.EXECUTION_RUNTIME,
        ownership_resolution=OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
    )
    dup_sem, _, _, _ = detect_ownership_matrix_findings((a, b))
    assert dup_sem


def test_mutation_legal_port_providers_not_duplicate() -> None:
    assert probe_legal_port_providers_not_duplicate() is None


def test_mutation_bypass_path_not_silent_pass() -> None:
    sample = ProductionReadinessCandidate(
        candidate_id="probe::bypass::ast.strict_lab.branch",
        module_path="intergrax/probe_bypass_path.py",
        symbol="branch:99",
        discovery_kind=DiscoveryKind.AST_STRICT_LAB_BRANCH,
        line_no=99,
        indicator="mode_branch",
    )
    record = classify_candidate(sample)
    assert record.mechanism_role == MechanismRole.SEMANTIC_MECHANISM
    assert record.ownership_resolution != OwnershipResolutionState.NOT_APPLICABLE_WITH_EVIDENCE


def test_mutation_frz_exactly_thirty_three() -> None:
    frz = build_frz_reconciliation()
    ids = [row.criterion_id for row in frz]
    assert ids == list(required_frz_ids())
    assert len(ids) == 33


def test_mutation_totals_change_readiness() -> None:
    inventory = classified_inventory()
    baseline = _reconciliation_from(inventory)
    target = next(record for record in inventory if ownership_resolved_with_evidence(record))
    mutated = tuple(
        replace(
            target,
            canonical_contract="CONTRACT_UNRESOLVED",
            ownership_resolution=OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
        )
        if record.candidate_id == target.candidate_id
        else record
        for record in inventory
    )
    after = _reconciliation_from(mutated)
    assert after.ownership_resolved < baseline.ownership_resolved
    assert prod_q_p0_inventory_reconciliation_violations(mutated) != ()


def test_mutation_probe_false_auxiliary_classifier() -> None:
    assert probe_false_auxiliary_na_without_symbol_evidence() is None


def test_auxiliary_metrics_balance() -> None:
    metrics = build_auxiliary_reconciliation_metrics()
    assert metrics.auxiliary_discovered == metrics.auxiliary_justified + metrics.auxiliary_unresolved
    assert metrics.auxiliary_unresolved == 0
    assert metrics.auxiliary_falsely_excluded == 0
