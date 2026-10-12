# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0 inventory identity, evidence, and mutation integrity gates."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from tests.qualification.prod_q._p0_classification import classify_candidate
from tests.qualification.prod_q._p0_discovery import raw_discovery_candidates
from tests.qualification.prod_q._p0_ownership import (
    _reconciliation_from,
    ownership_incomplete_reason,
    ownership_resolved_with_evidence,
)
from tests.qualification.prod_q._p0_types import (
    CandidateInventoryStatus,
    DiscoveryKind,
    MechanismRole,
    OwnershipResolutionState,
    ProductionReadinessCandidate,
    ProductionReadinessClassification,
    ProductionReadinessEvidenceState,
    ProductionReadinessFamily,
    ProductionReadinessOwner,
)


def inventory_identity_violations(
    raw: tuple[ProductionReadinessCandidate, ...],
    by_id: dict[str, ProductionReadinessClassification],
) -> tuple[str, ...]:
    violations: list[str] = []
    discovered = frozenset(c.candidate_id for c in raw)
    classified = frozenset(by_id)
    if discovered != classified:
        missing = sorted(discovered - classified)
        extra = sorted(classified - discovered)
        if missing:
            violations.append(f"missing classifications for discovered ids: {missing[:5]}")
        if extra:
            violations.append(f"classifications without discovery ids: {extra[:5]}")
    dupes = [cid for cid, count in Counter(c.candidate_id for c in raw).items() if count > 1]
    if dupes:
        violations.append(f"duplicate discovered candidate ids: {dupes[:5]}")
    for cid, record in by_id.items():
        if record.candidate_id != cid:
            violations.append(f"classification candidate_id mismatch for map key {cid}")
    return tuple(violations)


def unclassified_indicator_ids(
    by_id: dict[str, ProductionReadinessClassification],
) -> frozenset[str]:
    return frozenset(
        cid
        for cid, record in by_id.items()
        if record.inventory_status == CandidateInventoryStatus.UNCLASSIFIED_INDICATOR
    )


def assert_inventory_identity_sound(
    raw: tuple[ProductionReadinessCandidate, ...],
    by_id: dict[str, ProductionReadinessClassification],
) -> None:
    violations = inventory_identity_violations(raw, by_id)
    assert violations == (), "; ".join(violations)


def probe_unknown_indicator_classified_not_pq_f01() -> str | None:
    sample = next(iter(raw_discovery_candidates()))
    mutant = replace(sample, indicator="__prod_q_unknown_indicator__")
    record = classify_candidate(mutant)
    if record.family == ProductionReadinessFamily.PQ_F01:
        return "unknown indicator mapped to PQ-F01"
    if record.family != ProductionReadinessFamily.PQ_UNCLASSIFIED:
        return "unknown indicator not assigned PQ-UNCLASSIFIED family"
    if record.inventory_status != CandidateInventoryStatus.UNCLASSIFIED_INDICATOR:
        return "unknown indicator not marked UNCLASSIFIED_INDICATOR"
    if record.evidence_state != ProductionReadinessEvidenceState.BLOCKER:
        return "unknown indicator did not produce BLOCKER evidence_state"
    return None


def probe_fake_semantic_owner_not_resolved() -> str | None:
    sample = ProductionReadinessCandidate(
        candidate_id="probe::fake::ast.symbol.indicator",
        module_path="intergrax/probe_fake.py",
        symbol="probe_symbol",
        discovery_kind=DiscoveryKind.AST_SYMBOL_INDICATOR,
        line_no=1,
        indicator="health",
    )
    record = classify_candidate(sample)
    if record.ownership_resolution == OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE:
        return "mechanical discovery marked RESOLVED_WITH_CURRENT_HEAD_EVIDENCE"
    if record.semantic_owner == ProductionReadinessOwner.DISCOVERED_SURFACE and (
        record.ownership_resolution
        not in (
            OwnershipResolutionState.DISCOVERED,
            OwnershipResolutionState.NOT_YET_RESOLVED,
            OwnershipResolutionState.NOT_APPLICABLE_WITH_EVIDENCE,
        )
    ):
        return "DISCOVERED_SURFACE without expected non-resolved state"
    return None


def probe_removed_classification_detected(
    by_id: dict[str, ProductionReadinessClassification],
) -> str | None:
    if not by_id:
        return "empty inventory"
    first_key = next(iter(by_id))
    trimmed = dict(by_id)
    del trimmed[first_key]
    violations = inventory_identity_violations(raw_discovery_candidates(), trimmed)
    if not violations:
        return "removed classification not detected"
    return None


def probe_unresolved_contract_marks_incomplete() -> str | None:
    sample = ProductionReadinessCandidate(
        candidate_id="probe::unresolved_contract::anchor.registry",
        module_path="intergrax/probe_unresolved.py",
        symbol="ProbeSemantic",
        discovery_kind=DiscoveryKind.ANCHOR_REGISTRY,
        line_no=1,
        indicator="health",
    )
    record = classify_candidate(sample)
    if record.mechanism_role != MechanismRole.SEMANTIC_MECHANISM:
        return "anchor candidate not classified as semantic mechanism"
    if ownership_resolved_with_evidence(record):
        return "unresolved contract marked as ownership resolved"
    if ownership_incomplete_reason(record) != "unresolved_canonical_contract":
        return "expected unresolved_canonical_contract reason"
    return None


def probe_unresolved_composition_marks_incomplete() -> str | None:
    from dataclasses import replace

    from tests.qualification.prod_q._p0_classification import classification_by_candidate_id

    base = next(iter(classification_by_candidate_id().values()))
    mutant = replace(
        base,
        mechanism_role=MechanismRole.SEMANTIC_MECHANISM,
        canonical_contract="intergrax.probe.resolved_contract",
        composition_owner="COMPOSITION_UNRESOLVED",
        ownership_resolution=OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
        semantic_owner=ProductionReadinessOwner.EXECUTION_RUNTIME,
    )
    if ownership_resolved_with_evidence(mutant):
        return "unresolved composition counted as resolved"
    if ownership_incomplete_reason(mutant) != "unresolved_composition_owner":
        return "expected unresolved_composition_owner reason"
    return None


def probe_mutated_owner_alters_reconciliation() -> str | None:
    from dataclasses import replace

    from tests.qualification.prod_q._p0_classification import classified_inventory

    inventory = classified_inventory()
    baseline = _reconciliation_from(inventory)
    target = next(r for r in inventory if ownership_resolved_with_evidence(r))
    mutated = tuple(
        replace(
            target,
            canonical_contract="CONTRACT_UNRESOLVED",
            ownership_resolution=OwnershipResolutionState.NOT_YET_RESOLVED,
        )
        if r.candidate_id == target.candidate_id
        else r
        for r in inventory
    )
    after = _reconciliation_from(mutated)
    if baseline.ownership_resolved <= after.ownership_resolved:
        return "mutated owner/contract state did not reduce resolved ownership count"
    return None


def probe_readiness_method_not_auto_auxiliary() -> str | None:
    sample = ProductionReadinessCandidate(
        candidate_id="probe::readiness::ast.method.indicator",
        module_path="intergrax/probe_readiness.py",
        symbol="check_production_readiness",
        discovery_kind=DiscoveryKind.AST_METHOD_INDICATOR,
        line_no=1,
        indicator="readiness",
    )
    record = classify_candidate(sample)
    if record.mechanism_role == MechanismRole.AUXILIARY_DISCOVERY:
        return "operational readiness discovery classified as auxiliary"
    if record.mechanism_role != MechanismRole.SEMANTIC_MECHANISM:
        return "operational readiness discovery not classified as semantic mechanism"
    if record.ownership_resolution == OwnershipResolutionState.NOT_APPLICABLE_WITH_EVIDENCE:
        return "operational readiness discovery auto N/A"
    return None


def probe_false_auxiliary_na_without_symbol_evidence() -> str | None:
    sample = ProductionReadinessCandidate(
        candidate_id="probe::fake_na::ast.symbol.indicator",
        module_path="intergrax/probe_fake_na.py",
        symbol="incidental_mention",
        discovery_kind=DiscoveryKind.AST_SYMBOL_INDICATOR,
        line_no=1,
        indicator="lifecycle",
    )
    record = classify_candidate(sample)
    if record.mechanism_role != MechanismRole.AUXILIARY_DISCOVERY:
        return None
    if any(ref.startswith("auxiliary_na:symbol:") for ref in record.current_head_evidence):
        return None
    return "auxiliary role without symbol-scoped evidence not rejected by classifier"


def probe_unresolved_classification_role_explicit() -> str | None:
    sample = ProductionReadinessCandidate(
        candidate_id="probe::unresolved::role",
        module_path="intergrax/probe_unresolved.py",
        symbol="opaque_surface",
        discovery_kind=DiscoveryKind.AST_SYMBOL_INDICATOR,
        line_no=1,
        indicator="__opaque_token_without_family_mapping__",
    )
    record = classify_candidate(sample)
    if record.mechanism_role != MechanismRole.UNRESOLVED_CLASSIFICATION:
        return "unknown indicator token did not yield UNRESOLVED_CLASSIFICATION"
    if record.inventory_status != CandidateInventoryStatus.UNCLASSIFIED_INDICATOR:
        return "unknown indicator not marked UNCLASSIFIED_INDICATOR"
    return None


def probe_mutated_alternate_mechanism_detected() -> str | None:
    from tests.qualification.prod_q._p0_ownership import alternate_composition_violations

    from tests.qualification.prod_q._p0_classification import classified_inventory

    base = next(
        r
        for r in classified_inventory()
        if r.mechanism_role == MechanismRole.SEMANTIC_MECHANISM
        and r.ownership_resolution == OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE
    )
    shadow = replace(
        base,
        candidate_id="probe::alternate::mechanism",
        module="intergrax/probe/alternate_path.py",
        composition_owner="intergrax/probe/alternate_path.py",
    )
    if not alternate_composition_violations((base, shadow)):
        return "unsanctioned alternate composition path not detected"
    return None


def probe_duplicate_candidate_id_detected() -> str | None:
    raw = raw_discovery_candidates()
    if not raw:
        return "insufficient raw candidates"
    duped = (raw[0], raw[0])
    by_id = {raw[0].candidate_id: classify_candidate(raw[0])}
    violations = inventory_identity_violations(duped, by_id)
    if not any("duplicate" in v for v in violations):
        return "duplicate candidate id not detected"
    return None
