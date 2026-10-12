# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0 ownership reconciliation, concern identity, and soundness gates."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.prod_q._p0_classification import classified_inventory
from tests.qualification.prod_q._p0_findings_register import known_prod_q_findings
from tests.qualification.prod_q._p0_types import (
    MechanismRole,
    OwnershipReconciliationReport,
    OwnershipResolutionState,
    ProdQFindingDisposition,
    ProductionReadinessClassification,
    ProductionReadinessOwner,
)

_UNRESOLVED_CONTRACT_MARKERS = frozenset({"CONTRACT_UNRESOLVED", ""})
_UNRESOLVED_COMPOSITION_MARKERS = frozenset({"COMPOSITION_UNRESOLVED", ""})


def is_auxiliary_discovery(record: ProductionReadinessClassification) -> bool:
    if record.mechanism_role != MechanismRole.AUXILIARY_DISCOVERY:
        return False
    return any(ref.startswith("auxiliary_na:symbol:") for ref in record.current_head_evidence)


def is_semantic_mechanism(record: ProductionReadinessClassification) -> bool:
    return record.mechanism_role == MechanismRole.SEMANTIC_MECHANISM


def is_unresolved_classification(record: ProductionReadinessClassification) -> bool:
    return record.mechanism_role == MechanismRole.UNRESOLVED_CLASSIFICATION


def ownership_resolved_with_evidence(record: ProductionReadinessClassification) -> bool:
    if not is_semantic_mechanism(record):
        return False
    if record.ownership_resolution != OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE:
        return False
    if record.semantic_owner == ProductionReadinessOwner.DISCOVERED_SURFACE:
        return False
    if record.canonical_contract in _UNRESOLVED_CONTRACT_MARKERS:
        return False
    if record.composition_owner in _UNRESOLVED_COMPOSITION_MARKERS:
        return False
    return True


def concern_identity_for_ownership(record: ProductionReadinessClassification) -> str:
    """Concern-level identity for duplicate/alternate-contract detection (not candidate_id)."""
    if is_auxiliary_discovery(record) or is_unresolved_classification(record):
        return record.semantic_concern_key
    key = record.semantic_concern_key
    if key.startswith("concern:operation:") or key.startswith("concern:import:"):
        return key
    if not record.canonical_contract.startswith("CONTRACT_"):
        return f"concern:operation:contract:{record.canonical_contract}"
    if key.startswith("concern:unresolved:"):
        return key
    return (
        f"concern:operation:{record.family.value}:{record.responsibility.split(':', 1)[-1]}"
    )


def semantic_concern_alt_contract_key(record: ProductionReadinessClassification) -> str | None:
    if not ownership_resolved_with_evidence(record):
        return None
    return (
        f"concern:alt:owner:{record.semantic_owner.value}:"
        f"{concern_identity_for_ownership(record)}"
    )


def ownership_incomplete_reason(record: ProductionReadinessClassification) -> str | None:
    if not is_semantic_mechanism(record):
        return None
    if ownership_resolved_with_evidence(record):
        return None
    if record.canonical_contract in _UNRESOLVED_CONTRACT_MARKERS:
        return "unresolved_canonical_contract"
    if record.composition_owner in _UNRESOLVED_COMPOSITION_MARKERS:
        return "unresolved_composition_owner"
    if record.ownership_resolution == OwnershipResolutionState.AMBIGUOUS:
        return "ambiguous"
    if record.ownership_resolution == OwnershipResolutionState.NOT_YET_RESOLVED:
        return "not_yet_resolved"
    if record.ownership_resolution == OwnershipResolutionState.DISCOVERED:
        return "discovered_without_resolution"
    return "ownership_incomplete"


def detect_ownership_matrix_findings(
    inventory: tuple[ProductionReadinessClassification, ...],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    semantic = [r for r in inventory if is_semantic_mechanism(r)]
    by_concern: dict[str, list[ProductionReadinessClassification]] = {}
    for record in semantic:
        key = concern_identity_for_ownership(record)
        by_concern.setdefault(key, []).append(record)

    duplicate_semantic: list[str] = []
    duplicate_composition: list[str] = []
    unresolved: list[str] = []
    alt_contracts: list[str] = []
    alt_groups: dict[str, set[str]] = {}

    for key, items in sorted(by_concern.items()):
        owners = {item.semantic_owner for item in items}
        compositions = {item.composition_owner for item in items}
        contracts = {item.canonical_contract for item in items}
        if len(owners) > 1:
            duplicate_semantic.append(key)
        if len(compositions) > 1 and ProductionReadinessOwner.DISCOVERED_SURFACE not in owners:
            if len(owners) > 1 or len(contracts) > 1:
                duplicate_composition.append(key)
        for item in items:
            reason = ownership_incomplete_reason(item)
            if reason is not None:
                unresolved.append(f"{key}:{reason}")
        for item in items:
            alt_key = semantic_concern_alt_contract_key(item)
            if alt_key is None:
                continue
            alt_groups.setdefault(alt_key, set()).add(item.canonical_contract)

    for alt_key, contracts in alt_groups.items():
        if len(contracts) > 1:
            alt_contracts.append(alt_key)

    return (
        tuple(sorted(frozenset(duplicate_semantic))),
        tuple(sorted(frozenset(duplicate_composition))),
        tuple(sorted(frozenset(alt_contracts))),
        tuple(sorted(frozenset(unresolved))),
    )


@lru_cache(maxsize=1)
def build_ownership_reconciliation() -> OwnershipReconciliationReport:
    inventory = classified_inventory()
    candidate_count = len(inventory)
    semantic_records = [r for r in inventory if is_semantic_mechanism(r)]
    auxiliary_count = sum(1 for r in inventory if is_auxiliary_discovery(r))
    unresolved_class_count = sum(1 for r in inventory if is_unresolved_classification(r))

    resolved = 0
    unresolved = 0
    ambiguous = 0
    na_with_evidence = 0

    for record in inventory:
        if is_auxiliary_discovery(record):
            if record.ownership_resolution == OwnershipResolutionState.NOT_APPLICABLE_WITH_EVIDENCE:
                na_with_evidence += 1
            continue
        if is_unresolved_classification(record):
            continue
        reason = ownership_incomplete_reason(record)
        if reason == "ambiguous":
            ambiguous += 1
        elif reason is None:
            resolved += 1
        else:
            unresolved += 1

    dup_sem, dup_comp, alt, unres_keys = detect_ownership_matrix_findings(tuple(inventory))

    return OwnershipReconciliationReport(
        candidate_count=candidate_count,
        semantic_mechanism_count=len(semantic_records),
        auxiliary_discovery_count=auxiliary_count,
        unresolved_classification_count=unresolved_class_count,
        ownership_resolved=resolved,
        ownership_unresolved=unresolved,
        ownership_ambiguous=ambiguous,
        ownership_not_applicable_with_evidence=na_with_evidence,
        duplicate_semantic_concern_keys=dup_sem,
        duplicate_composition_concern_keys=dup_comp,
        alternate_canonical_contract_concern_keys=alt,
        unresolved_semantic_concern_keys=unres_keys,
    )


def ownership_soundness_violations(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> tuple[str, ...]:
    inv = inventory if inventory is not None else classified_inventory()
    violations: list[str] = []
    for record in inv:
        if not is_semantic_mechanism(record):
            continue
        if ownership_resolved_with_evidence(record):
            continue
        if record.ownership_resolution == OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE:
            violations.append(
                f"{record.candidate_id}: marked RESOLVED but contract/owner/composition not evidenced"
            )
        if record.semantic_owner == ProductionReadinessOwner.DISCOVERED_SURFACE and (
            record.ownership_resolution == OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE
        ):
            violations.append(f"{record.candidate_id}: DISCOVERED_SURFACE counted as resolved")
    recon = build_ownership_reconciliation() if inventory is None else _reconciliation_from(inv)
    if recon.duplicate_semantic_concern_keys:
        violations.append(
            f"duplicate semantic concern owners: {recon.duplicate_semantic_concern_keys[:3]}"
        )
    return tuple(violations)


def _reconciliation_from(inventory: tuple[ProductionReadinessClassification, ...]) -> OwnershipReconciliationReport:
    candidate_count = len(inventory)
    semantic_records = [r for r in inventory if is_semantic_mechanism(r)]
    auxiliary_count = sum(1 for r in inventory if is_auxiliary_discovery(r))
    unresolved_class_count = sum(1 for r in inventory if is_unresolved_classification(r))
    resolved = unresolved = ambiguous = na_with_evidence = 0
    for record in inventory:
        if is_auxiliary_discovery(record):
            if record.ownership_resolution == OwnershipResolutionState.NOT_APPLICABLE_WITH_EVIDENCE:
                na_with_evidence += 1
            continue
        if is_unresolved_classification(record):
            continue
        reason = ownership_incomplete_reason(record)
        if reason == "ambiguous":
            ambiguous += 1
        elif reason is None:
            resolved += 1
        else:
            unresolved += 1
    dup_sem, dup_comp, alt, unres_keys = detect_ownership_matrix_findings(inventory)
    return OwnershipReconciliationReport(
        candidate_count=candidate_count,
        semantic_mechanism_count=len(semantic_records),
        auxiliary_discovery_count=auxiliary_count,
        unresolved_classification_count=unresolved_class_count,
        ownership_resolved=resolved,
        ownership_unresolved=unresolved,
        ownership_ambiguous=ambiguous,
        ownership_not_applicable_with_evidence=na_with_evidence,
        duplicate_semantic_concern_keys=dup_sem,
        duplicate_composition_concern_keys=dup_comp,
        alternate_canonical_contract_concern_keys=alt,
        unresolved_semantic_concern_keys=unres_keys,
    )


_SANCTIONED_ALTERNATE_COMPOSITION_KEYS = frozenset(
    item.concern_key
    for item in known_prod_q_findings()
    if item.disposition == ProdQFindingDisposition.LEGAL_DISTINCT_COMPOSITION
)


def alternate_composition_violations(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> tuple[str, ...]:
    inv = inventory if inventory is not None else classified_inventory()
    semantic = [r for r in inv if is_semantic_mechanism(r)]
    by_concern: dict[str, list[ProductionReadinessClassification]] = {}
    for record in semantic:
        key = concern_identity_for_ownership(record)
        by_concern.setdefault(key, []).append(record)

    violations: list[str] = []
    for key, items in by_concern.items():
        compositions = {item.composition_owner for item in items}
        owners = {item.semantic_owner for item in items}
        contracts = {item.canonical_contract for item in items}
        if len(compositions) <= 1:
            continue
        if ProductionReadinessOwner.DISCOVERED_SURFACE in owners:
            continue
        if len(owners) == 1 and len(contracts) == 1 and key in _SANCTIONED_ALTERNATE_COMPOSITION_KEYS:
            continue
        violations.append(f"unsanctioned alternate composition for {key}")
    return tuple(violations)


def auxiliary_na_without_symbol_evidence(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> tuple[str, ...]:
    inv = inventory if inventory is not None else classified_inventory()
    return tuple(
        f"{record.candidate_id}: auxiliary without auxiliary_na:symbol evidence"
        for record in inv
        if record.mechanism_role == MechanismRole.AUXILIARY_DISCOVERY
        and not any(ref.startswith("auxiliary_na:symbol:") for ref in record.current_head_evidence)
    )


def unresolved_role_classifications(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> tuple[str, ...]:
    """Fail-closed when UNRESOLVED_CLASSIFICATION appears on current HEAD (must be 0 for READY)."""
    inv = inventory if inventory is not None else classified_inventory()
    unresolved = [r for r in inv if is_unresolved_classification(r)]
    if not unresolved:
        return ()
    return (
        f"unresolved mechanism role classifications on current HEAD: {len(unresolved)}",
    )


def probe_alternate_contract_conflict_detected() -> str | None:
    from dataclasses import replace

    from tests.qualification.prod_q._p0_types import OwnershipResolutionState

    base = next(
        r
        for r in classified_inventory()
        if is_semantic_mechanism(r) and ownership_resolved_with_evidence(r)
    )
    operation_key = f"concern:operation:{base.family.value}:health:shared_probe"
    a = replace(
        base,
        candidate_id="probe::alt::a",
        canonical_contract="intergrax.probe.contract.A",
        module="intergrax/probe/module_a.py",
        symbol="shared_semantic_responsibility",
        semantic_owner=ProductionReadinessOwner.HOSTING_ENGINE,
        semantic_concern_key=operation_key,
        composition_owner="intergrax/probe/module_a.py",
        ownership_resolution=OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
    )
    b = replace(
        base,
        candidate_id="probe::alt::b",
        canonical_contract="intergrax.probe.contract.B",
        module="intergrax/probe/module_b.py",
        symbol="shared_semantic_responsibility",
        semantic_owner=ProductionReadinessOwner.EXECUTION_RUNTIME,
        semantic_concern_key=operation_key,
        composition_owner="intergrax/probe/module_b.py",
        ownership_resolution=OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
    )
    dup_sem, _, _, _ = detect_ownership_matrix_findings((a, b))
    if not dup_sem:
        return "cross-module cross-contract duplicate semantic owners not detected"
    return None


def probe_legal_port_providers_not_duplicate() -> str | None:
    from tests.qualification.prod_q._p0_classification import classified_inventory

    inventory = classified_inventory()
    key = "concern:operation:contract:intergrax.contracts.execution_capacity_admission.ExecutionCapacityAdmissionPort"
    items = [
        r
        for r in inventory
        if is_semantic_mechanism(r) and concern_identity_for_ownership(r) == key
    ]
    if len(items) < 2:
        return None
    dup_sem, dup_comp, _, _ = detect_ownership_matrix_findings(tuple(items))
    if dup_sem or dup_comp:
        return "legal implementation providers behind one contract flagged as duplicate"
    return None


def probe_distinct_concerns_not_merged() -> str | None:
    from tests.qualification.prod_q._p0_types import ProductionReadinessFamily

    semantic = [r for r in classified_inventory() if is_semantic_mechanism(r)]
    health = [r for r in semantic if r.family == ProductionReadinessFamily.PQ_F04]
    if len(health) < 2:
        return None
    keys = {concern_identity_for_ownership(r) for r in health}
    if len(keys) < 2:
        return "distinct health semantic mechanisms collapsed to one concern identity"
    return None


def probe_concern_identity_stable_under_candidate_order() -> str | None:
    from tests.qualification.prod_q._p0_classification import classify_candidate
    from tests.qualification.prod_q._p0_discovery import raw_discovery_candidates

    raw = raw_discovery_candidates()
    fwd = {c.candidate_id: classify_candidate(c).semantic_concern_key for c in raw}
    rev = {c.candidate_id: classify_candidate(c).semantic_concern_key for c in reversed(raw)}
    if fwd != rev:
        return "semantic_concern_key depends on discovery iteration order"
    return None
