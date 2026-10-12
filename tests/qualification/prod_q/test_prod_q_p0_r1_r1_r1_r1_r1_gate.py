# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0-R1-R1-R1-R1-R1 evidence integrity and fail-closed disposition closure."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.qualification.prod_q._p0_audit_gate import (
    PROD_Q_P0_R1_R1_R1_R1_R1_STAGE_ID,
    prod_q_p0_inventory_reconciliation_violations,
    prod_q_p0_r1_r1_r1_r1_r1_gate_violations,
    prod_q_p0_r1_r1_r1_r1_r1_ready_for_audit,
)
from tests.qualification.prod_q._p0_classification import classified_inventory, classify_candidate
from tests.qualification.prod_q._p0_findings_register import (
    build_reconciled_prod_q_findings,
    build_semantic_concern_records,
    concern_finding_reconciliation_violations,
    disposition_for_unresolved_concern,
    historical_prod_q_findings,
)
from tests.qualification.prod_q._p0_ownership import (
    apply_auxiliary_finalization,
    auxiliary_evidence_is_sufficient,
    build_auxiliary_justifications,
    concern_identity_for_ownership,
    is_auxiliary_discovery,
)
from tests.qualification.prod_q._p0_types import (
    AuxiliaryEvidenceStrength,
    AuxiliaryJustification,
    AuxiliaryJustificationKind,
    AuxiliaryVerificationResult,
    DiscoveryKind,
    MechanismRole,
    OwnershipResolutionState,
    ProdQFindingDisposition,
    ProductionReadinessCandidate,
    ProductionReadinessEvidenceState,
    ProductionReadinessFamily,
    ProductionReadinessOwner,
    SemanticConcernResolutionState,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_prod_q_p0_r1_r1_r1_r1_r1_stage_id() -> None:
    assert PROD_Q_P0_R1_R1_R1_R1_R1_STAGE_ID == "PROD-Q-P0-R1-R1-R1-R1-R1"


def test_prod_q_p0_r1_r1_r1_r1_r1_ready_for_audit_gate() -> None:
    assert prod_q_p0_r1_r1_r1_r1_r1_ready_for_audit() is True
    assert prod_q_p0_r1_r1_r1_r1_r1_gate_violations() == ()


def test_auxiliary_same_module_hint_not_na() -> None:
    raw = (
        ProductionReadinessCandidate(
            candidate_id="probe::aux::same_module",
            module_path="intergrax/probe_aux_same_module.py",
            symbol="helper_metric",
            discovery_kind=DiscoveryKind.AST_SYMBOL_INDICATOR,
            line_no=1,
            indicator="lifecycle",
        ),
    )
    preliminary = tuple(classify_candidate(candidate) for candidate in raw)
    justifications = build_auxiliary_justifications(raw, preliminary)
    record = apply_auxiliary_finalization(preliminary[0], justifications[preliminary[0].candidate_id])
    assert record.mechanism_role == MechanismRole.SEMANTIC_MECHANISM
    assert record.evidence_state != ProductionReadinessEvidenceState.N_A_WITH_EVIDENCE


def test_auxiliary_family_hint_not_na() -> None:
    inventory = classified_inventory()
    hinted = next(
        record
        for record in inventory
        if any(ref == "auxiliary_rationale:co_classified_family_semantic_anchor" for ref in record.current_head_evidence)
        or any(ref == "auxiliary_reclassified:unverified_relationship" for ref in record.current_head_evidence)
    )
    assert hinted.mechanism_role == MechanismRole.SEMANTIC_MECHANISM


def test_auxiliary_same_class_hint_not_na() -> None:
    inventory = classified_inventory()
    hinted = next(
        record
        for record in inventory
        if any(ref == "auxiliary_rationale:same_enclosing_class_hint_only" for ref in record.current_head_evidence)
    )
    assert not is_auxiliary_discovery(hinted)


def test_auxiliary_contract_import_hint_not_na() -> None:
    raw = (
        ProductionReadinessCandidate(
            candidate_id="probe::aux::contract_import",
            module_path="intergrax/contracts/execution_capacity_admission.py",
            symbol="unrelated_symbol",
            discovery_kind=DiscoveryKind.AST_SYMBOL_INDICATOR,
            line_no=1,
            indicator="lifecycle",
        ),
    )
    preliminary = tuple(classify_candidate(candidate) for candidate in raw)
    justifications = build_auxiliary_justifications(raw, preliminary)
    record = apply_auxiliary_finalization(preliminary[0], justifications[preliminary[0].candidate_id])
    assert not auxiliary_evidence_is_sufficient(record)


def test_auxiliary_unreachable_hint_not_universal_exclusion() -> None:
    inventory = classified_inventory()
    hinted = next(
        record
        for record in inventory
        if any(
            ref == "relationship_evidence:reachability:universal_exclusion_not_proven"
            for ref in record.current_head_evidence
        )
    )
    assert hinted.evidence_state != ProductionReadinessEvidenceState.N_A_WITH_EVIDENCE


def test_auxiliary_verified_private_nested_eligible_for_na() -> None:
    inventory = classified_inventory()
    verified = next(record for record in inventory if is_auxiliary_discovery(record))
    assert any(ref.startswith("verification_result:verified") for ref in verified.current_head_evidence)
    assert auxiliary_evidence_is_sufficient(verified)


def test_auxiliary_loses_related_concern_gate_blocks() -> None:
    inventory = classified_inventory()
    target = next(record for record in inventory if is_auxiliary_discovery(record))
    stripped = replace(
        target,
        current_head_evidence=tuple(
            ref
            for ref in target.current_head_evidence
            if not ref.startswith(("related_concern:", "verification_result:verified"))
        ),
    )
    mutated = tuple(stripped if record.candidate_id == target.candidate_id else record for record in inventory)
    assert prod_q_p0_inventory_reconciliation_violations(mutated)


def test_unknown_concern_not_default_freeze_debt() -> None:
    disposition, child = disposition_for_unresolved_concern(
        "concern:unresolved:semantic:PQ-UNCLASSIFIED:probe:unknown",
        ProductionReadinessFamily.PQ_UNCLASSIFIED,
        (),
    )
    assert disposition == ProdQFindingDisposition.UNRESOLVED_DISPOSITION
    assert child is None


def test_blocker_masked_as_debt_gate_blocks() -> None:
    inventory = classified_inventory()
    concerns = build_semantic_concern_records(inventory)
    target = next(
        concern
        for concern in concerns
        if concern.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER
    )
    findings = list(build_reconciled_prod_q_findings(inventory))
    finding = next(item for item in findings if item.concern_key == target.concern_key)
    masked = replace(finding, disposition=ProdQFindingDisposition.TRACKED_FREEZE_DEBT)
    findings = [masked if item.concern_key == target.concern_key else item for item in findings]
    assert concern_finding_reconciliation_violations(inventory, tuple(findings))


def test_legitimate_freeze_debt_with_child_allowed() -> None:
    inventory = classified_inventory()
    concerns = build_semantic_concern_records(inventory)
    debt = next(
        concern
        for concern in concerns
        if concern.finding_disposition == ProdQFindingDisposition.TRACKED_FREEZE_DEBT
        and concern.mandatory_child is not None
    )
    assert debt.resolution_state == SemanticConcernResolutionState.UNRESOLVED


def test_in_scope_blocker_requires_mandatory_child() -> None:
    inventory = classified_inventory()
    concerns = build_semantic_concern_records(inventory)
    blocker = next(
        concern
        for concern in concerns
        if concern.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER
    )
    findings = list(build_reconciled_prod_q_findings(inventory))
    finding = next(item for item in findings if item.concern_key == blocker.concern_key)
    broken = replace(finding, recommended_child=None)
    findings = [broken if item.concern_key == blocker.concern_key else item for item in findings]
    assert any(
        "missing mandatory child" in item
        for item in concern_finding_reconciliation_violations(inventory, tuple(findings))
    )


def test_historical_resolved_without_current_contract_unresolved() -> None:
    inventory = classified_inventory()
    base = next(record for record in inventory if record.ownership_resolution == OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE)
    concern_key = concern_identity_for_ownership(base)
    mutated = tuple(
        replace(
            base,
            canonical_contract="CONTRACT_UNRESOLVED",
            composition_owner="COMPOSITION_UNRESOLVED",
            ownership_resolution=OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
        )
        if record.candidate_id == base.candidate_id
        else record
        for record in inventory
    )
    concern = next(
        item
        for item in build_semantic_concern_records(mutated)
        if item.concern_key == concern_key
    )
    assert concern.resolution_state == SemanticConcernResolutionState.UNRESOLVED


def test_historical_legal_distinct_conflicting_composition_blocks() -> None:
    inventory = classified_inventory()
    key = "concern:operation:contract:intergrax.runtime.observability.auditability_health"
    members = [record for record in inventory if record.semantic_concern_key.startswith("concern:operation")]
    base = members[0]
    conflict = replace(
        base,
        candidate_id="probe::legal_distinct_conflict",
        semantic_concern_key=key,
        canonical_contract="intergrax.probe.conflicting.auditability_health",
        composition_owner="intergrax/probe/conflict.py",
        semantic_owner=ProductionReadinessOwner.OBSERVABILITY_AUDITABILITY,
        ownership_resolution=OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
    )
    mutated = inventory + (conflict,)
    concern = next(item for item in build_semantic_concern_records(mutated) if item.concern_key == key)
    assert concern.resolution_state != SemanticConcernResolutionState.LEGAL_DISTINCT_COMPOSITION


def test_historical_lineage_readable_after_correction() -> None:
    concerns = build_semantic_concern_records()
    lineage = [
        concern
        for concern in concerns
        if any(ref.startswith("historical_lineage:") for ref in concern.source_evidence)
    ]
    assert lineage
    assert len(historical_prod_q_findings()) == 8


def test_valid_current_resolved_concern_passes() -> None:
    concerns = build_semantic_concern_records()
    resolved = [
        concern
        for concern in concerns
        if concern.resolution_state == SemanticConcernResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE
    ]
    assert resolved
    assert concern_finding_reconciliation_violations() == ()


def test_inject_false_auxiliary_na_blocks_ready() -> None:
    inventory = classified_inventory()
    target = next(record for record in inventory if is_auxiliary_discovery(record))
    fake = replace(
        target,
        current_head_evidence=target.current_head_evidence + ("auxiliary_na:symbol:fake",),
    )
    mutated = tuple(fake if record.candidate_id == target.candidate_id else record for record in inventory)
    assert prod_q_p0_inventory_reconciliation_violations(mutated)


def test_inject_unknown_disposition_blocks_ready() -> None:
    inventory = classified_inventory()
    concerns = build_semantic_concern_records(inventory)
    target = next(concern for concern in concerns if concern.resolution_state == SemanticConcernResolutionState.UNRESOLVED)
    broken = replace(target, finding_disposition=ProdQFindingDisposition.UNRESOLVED_DISPOSITION, mandatory_child=None)
    findings = list(build_reconciled_prod_q_findings(inventory))
    findings.append(
        replace(
            findings[0],
            finding_id="PROD-Q-P0-UNRESOLVED-DISPOSITION-PROBE",
            concern_key=broken.concern_key,
            disposition=ProdQFindingDisposition.UNRESOLVED_DISPOSITION,
            recommended_child=None,
            lineage_only=False,
        )
    )
    assert concern_finding_reconciliation_violations(inventory, tuple(findings))


def test_inject_historical_false_resolved_blocks_ready() -> None:
    inventory = classified_inventory()
    concerns = build_semantic_concern_records(inventory)
    target = next(
        concern
        for concern in concerns
        if concern.resolution_state == SemanticConcernResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE
    )
    unresolved_members = tuple(
        replace(
            record,
            canonical_contract="CONTRACT_UNRESOLVED",
            ownership_resolution=OwnershipResolutionState.NOT_YET_RESOLVED,
        )
        if record.candidate_id in target.member_candidate_ids
        else record
        for record in inventory
    )
    rebuilt = build_semantic_concern_records(unresolved_members)
    broken = next(item for item in rebuilt if item.concern_key == target.concern_key)
    assert broken.resolution_state == SemanticConcernResolutionState.UNRESOLVED


def test_member_concern_identity_mutation_detected() -> None:
    inventory = classified_inventory()
    concerns = build_semantic_concern_records(inventory)
    target = next(
        concern
        for concern in concerns
        if concern.resolution_state == SemanticConcernResolutionState.UNRESOLVED
        and len(concern.member_candidate_ids) >= 2
    )
    drop_id = target.member_candidate_ids[0]
    mutated = tuple(record for record in inventory if record.candidate_id != drop_id)
    stale_findings = build_reconciled_prod_q_findings(inventory)
    assert concern_finding_reconciliation_violations(mutated, stale_findings) != ()


def test_restore_valid_evidence_restores_gate() -> None:
    inventory = classified_inventory()
    assert prod_q_p0_r1_r1_r1_r1_r1_gate_violations() == ()
    target = next(record for record in inventory if is_auxiliary_discovery(record))
    broken = replace(target, current_head_evidence=("discovery:ast.symbol.indicator",))
    mutated = tuple(broken if record.candidate_id == target.candidate_id else record for record in inventory)
    assert prod_q_p0_inventory_reconciliation_violations(mutated)
    assert prod_q_p0_r1_r1_r1_r1_r1_gate_violations() == ()


def test_verified_helper_justification_shape() -> None:
    justification = AuxiliaryJustification(
        candidate_id="probe::verified",
        kind=AuxiliaryJustificationKind.HELPER_OF_CLASSIFIED_CONCERN,
        related_concern_key="concern:operation:PQ-F04:health",
        source_location="intergrax/probe.py:Host._helper:L1",
        strength=AuxiliaryEvidenceStrength.PROVEN_STATIC,
        rationale="verified_private_nested_helper",
        source_symbol_identity="intergrax/probe.py::Host._helper",
        relationship_evidence=("private_method:_helper",),
        verification_result=AuxiliaryVerificationResult.VERIFIED,
    )
    assert justification.verification_result == AuxiliaryVerificationResult.VERIFIED
