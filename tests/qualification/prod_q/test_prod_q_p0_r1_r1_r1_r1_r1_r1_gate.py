# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0-R1-R1-R1-R1-R1-R1 auxiliary relationship proof and factory ownership integrity."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.qualification.prod_q._p0_audit_gate import (
    PROD_Q_P0_R1_R1_R1_R1_R1_R1_R1_STAGE_ID,
    prod_q_p0_inventory_reconciliation_violations,
    prod_q_p0_r1_r1_r1_r1_r1_r1_r1_gate_violations,
    prod_q_p0_r1_r1_r1_r1_r1_r1_r1_ready_for_audit,
)
from tests.qualification.prod_q._p0_classification import classified_inventory, classify_candidate
from tests.qualification.prod_q._p0_findings_register import (
    build_reconciled_prod_q_findings,
    build_semantic_concern_records,
    concern_finding_reconciliation_violations,
    factory_anchor_ownership_current_head_evidence,
    historical_prod_q_findings,
)
from tests.qualification.prod_q._p0_ownership import (
    apply_auxiliary_finalization,
    auxiliary_evidence_is_sufficient,
    build_auxiliary_justifications,
    is_auxiliary_discovery,
    probe_legal_port_providers_not_duplicate,
)
from tests.qualification.prod_q._p0_types import (
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


def test_prod_q_p0_r1_r1_r1_r1_r1_r1_r1_stage_id() -> None:
    assert PROD_Q_P0_R1_R1_R1_R1_R1_R1_R1_STAGE_ID == "PROD-Q-P0-R1-R1-R1-R1-R1-R1-R1"


def test_prod_q_p0_r1_r1_r1_r1_r1_r1_r1_ready_for_audit_gate() -> None:
    assert prod_q_p0_r1_r1_r1_r1_r1_r1_r1_ready_for_audit() is True
    assert prod_q_p0_r1_r1_r1_r1_r1_r1_r1_gate_violations() == ()


def test_private_helper_same_class_without_call_not_verified() -> None:
    raw = (
        ProductionReadinessCandidate(
            candidate_id="probe::aux::no_call",
            module_path="intergrax/runtime/observability/auditability_health.py",
            symbol="AuditabilityHealthSnapshotBuilder._unused_private_helper",
            discovery_kind=DiscoveryKind.AST_SYMBOL_INDICATOR,
            line_no=1,
            indicator="lifecycle",
        ),
    )
    preliminary = tuple(classify_candidate(candidate) for candidate in raw)
    justifications = build_auxiliary_justifications(raw, preliminary)
    record = apply_auxiliary_finalization(preliminary[0], justifications[preliminary[0].candidate_id])
    assert not auxiliary_evidence_is_sufficient(record)
    assert record.mechanism_role == MechanismRole.SEMANTIC_MECHANISM


def test_private_helper_with_independent_public_entry_not_verified() -> None:
    raw = (
        ProductionReadinessCandidate(
            candidate_id="probe::aux::public_entry",
            module_path="applications/local_workspace_application/workspaces/tenant_connection_product_orchestration.py",
            symbol="TenantConnectionProductOrchestrationService._read_secret",
            discovery_kind=DiscoveryKind.AST_SYMBOL_INDICATOR,
            line_no=1,
            indicator="lifecycle",
        ),
    )
    preliminary = tuple(classify_candidate(candidate) for candidate in raw)
    justifications = build_auxiliary_justifications(raw, preliminary)
    record = apply_auxiliary_finalization(preliminary[0], justifications[preliminary[0].candidate_id])
    assert not is_auxiliary_discovery(record)


def test_private_helper_with_proven_call_may_be_verified() -> None:
    inventory = classified_inventory()
    verified = [record for record in inventory if is_auxiliary_discovery(record)]
    assert verified, "expected at least one VERIFIED auxiliary on current HEAD"
    sample = verified[0]
    assert any(
        ref.startswith("relationship_evidence:caller_to_callee:")
        for ref in sample.current_head_evidence
    )
    assert auxiliary_evidence_is_sufficient(sample)


def test_removing_caller_callee_relationship_revokes_verified() -> None:
    inventory = classified_inventory()
    target = next(record for record in inventory if is_auxiliary_discovery(record))
    stripped = replace(
        target,
        current_head_evidence=tuple(
            ref
            for ref in target.current_head_evidence
            if not ref.startswith("relationship_evidence:caller_to_callee:")
        ),
    )
    mutated = tuple(stripped if record.candidate_id == target.candidate_id else record for record in inventory)
    assert prod_q_p0_inventory_reconciliation_violations(mutated)


def test_factory_name_path_without_contract_not_resolved() -> None:
    inventory = classified_inventory()
    factory = next(
        record
        for record in inventory
        if record.symbol == "create_e2b_sandbox_host_integration"
    )
    fake = replace(
        factory,
        symbol="create_unproven_sandbox_host_integration",
        canonical_contract="CONTRACT_UNRESOLVED",
        composition_owner="COMPOSITION_UNRESOLVED",
        ownership_resolution=OwnershipResolutionState.NOT_YET_RESOLVED,
        current_head_evidence=factory.current_head_evidence + ("discovery:anchor.registry",),
    )
    mutated = tuple(fake if record.candidate_id == factory.candidate_id else record for record in inventory)
    concerns = build_semantic_concern_records(mutated)
    anchor = next(
        c
        for c in concerns
        if c.concern_key == "concern:operation:PQ-F09:sandbox_provider:e2b:factory_anchor"
    )
    assert anchor.resolution_state == SemanticConcernResolutionState.UNRESOLVED


def test_factory_without_composition_owner_not_resolved() -> None:
    evidence = factory_anchor_ownership_current_head_evidence(())
    assert evidence is None


def test_factory_composition_unresolved_inventory_still_structurally_proven() -> None:
    factory = next(
        record
        for record in classified_inventory()
        if record.symbol == "create_e2b_sandbox_host_integration"
    )
    assert factory.composition_owner.endswith("/register.py")
    evidence = factory_anchor_ownership_current_head_evidence((factory,))
    assert evidence is not None
    assert "ownership_assertion:OWNERSHIP_RESOLVED" in evidence


def test_factory_with_current_contract_owner_composition_resolved() -> None:
    inventory = classified_inventory()
    factory = next(
        record
        for record in inventory
        if record.symbol == "create_modal_sandbox_host_integration"
    )
    evidence = factory_anchor_ownership_current_head_evidence((factory,))
    assert evidence is not None
    assert any(ref.startswith("canonical_contract:") for ref in evidence)
    assert any(ref.startswith("contract_declaration_owner:") for ref in evidence)
    concerns = build_semantic_concern_records(inventory)
    anchor = next(
        c
        for c in concerns
        if c.concern_key == "concern:operation:PQ-F09:sandbox_provider:modal:factory_anchor"
    )
    assert anchor.resolution_state == SemanticConcernResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE


def test_ownership_resolved_not_production_qualified() -> None:
    concerns = build_semantic_concern_records()
    anchor = next(
        c
        for c in concerns
        if c.concern_key == "concern:operation:PQ-F09:sandbox_provider:e2b:factory_anchor"
    )
    assert anchor.resolution_state == SemanticConcernResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE
    provider = next(
        c
        for c in concerns
        if c.concern_key == "concern:operation:PQ-F09:sandbox_provider:e2b"
    )
    assert provider.resolution_state == SemanticConcernResolutionState.UNRESOLVED
    assert provider.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER


def test_historical_resolved_current_unresolved_active_p1_blocker() -> None:
    inventory = classified_inventory()
    factory = next(
        record
        for record in inventory
        if record.symbol == "create_e2b_sandbox_host_integration"
    )
    key = "concern:operation:PQ-F09:sandbox_provider:e2b:factory_anchor"
    broken = replace(
        factory,
        module="intergrax/integrations/providers/sandbox_host/e2b/missing_bundle.py",
        ownership_resolution=OwnershipResolutionState.NOT_YET_RESOLVED,
        composition_owner="COMPOSITION_UNRESOLVED",
    )
    mutated = tuple(broken if record.candidate_id == factory.candidate_id else record for record in inventory)
    concerns = build_semantic_concern_records(mutated)
    concern = next(c for c in concerns if c.concern_key == key)
    assert concern.resolution_state == SemanticConcernResolutionState.UNRESOLVED
    findings = build_reconciled_prod_q_findings(mutated)
    active = next(f for f in findings if f.concern_key == key and not f.lineage_only)
    assert active.disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER
    assert active.recommended_child == "PROD-Q-P1"
    assert any(
        item.concern_key == key
        and item.disposition == ProdQFindingDisposition.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE
        for item in historical_prod_q_findings()
    )


def test_historical_finding_does_not_remove_current_coverage() -> None:
    inventory = classified_inventory()
    concerns = build_semantic_concern_records(inventory)
    unresolved = [c for c in concerns if c.resolution_state == SemanticConcernResolutionState.UNRESOLVED]
    findings = build_reconciled_prod_q_findings(inventory)
    active_keys = {f.concern_key for f in findings if not f.lineage_only}
    assert {c.concern_key for c in unresolved}.issubset(active_keys)


def test_historical_finding_no_duplicate_active_disposition() -> None:
    assert concern_finding_reconciliation_violations() == ()


def test_unknown_provider_factory_fail_closed() -> None:
    inventory = classified_inventory()
    unknown = replace(
        next(record for record in inventory if record.symbol.startswith("create_")),
        candidate_id="probe::factory::unknown",
        module="intergrax/integrations/providers/sandbox_host/unknown/bundle.py",
        symbol="create_unknown_sandbox_host_integration",
        semantic_concern_key="concern:operation:PQ-F09:sandbox_provider:unknown:factory_anchor",
        ownership_resolution=OwnershipResolutionState.NOT_YET_RESOLVED,
        canonical_contract="CONTRACT_UNRESOLVED",
        composition_owner="COMPOSITION_UNRESOLVED",
        current_head_evidence=("discovery:anchor.registry",),
    )
    mutated = inventory + (unknown,)
    assert factory_anchor_ownership_current_head_evidence((unknown,)) is None


def test_cross_contract_alternate_owner_still_detected() -> None:
    from tests.qualification.prod_q._p0_ownership import probe_alternate_contract_conflict_detected

    assert probe_alternate_contract_conflict_detected() is None


def test_legal_dual_provider_implementations_not_duplicate_owner() -> None:
    assert probe_legal_port_providers_not_duplicate() is None


def test_violations_1_5_9_flip_gate_to_blocked() -> None:
    inventory = classified_inventory()
    assert prod_q_p0_r1_r1_r1_r1_r1_r1_r1_gate_violations() == ()

    verified = next(record for record in inventory if is_auxiliary_discovery(record))
    stripped = replace(
        verified,
        current_head_evidence=tuple(
            ref
            for ref in verified.current_head_evidence
            if not ref.startswith("relationship_evidence:caller_to_callee:")
        ),
    )
    assert prod_q_p0_inventory_reconciliation_violations(
        tuple(stripped if record.candidate_id == verified.candidate_id else record for record in inventory)
    )

    factory = next(
        record
        for record in inventory
        if record.symbol == "create_daytona_sandbox_host_integration"
    )
    broken_factory = replace(
        factory,
        ownership_resolution=OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
        canonical_contract="CONTRACT_UNRESOLVED",
        composition_owner="COMPOSITION_UNRESOLVED",
    )
    mutated_factory = tuple(
        broken_factory if record.candidate_id == factory.candidate_id else record for record in inventory
    )
    assert prod_q_p0_inventory_reconciliation_violations(mutated_factory)

    concerns = build_semantic_concern_records(inventory)
    resolved = next(
        c
        for c in concerns
        if c.resolution_state == SemanticConcernResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE
        and not c.concern_key.endswith(":factory_anchor")
    )
    unresolved_members = tuple(
        replace(
            record,
            canonical_contract="CONTRACT_UNRESOLVED",
            ownership_resolution=OwnershipResolutionState.NOT_YET_RESOLVED,
        )
        if record.candidate_id in resolved.member_candidate_ids
        else record
        for record in inventory
    )
    stale_findings = tuple(
        f
        for f in build_reconciled_prod_q_findings(inventory)
        if f.concern_key != resolved.concern_key or f.lineage_only
    )
    violations = concern_finding_reconciliation_violations(unresolved_members, stale_findings)
    assert any("missing finding coverage" in item for item in violations)
