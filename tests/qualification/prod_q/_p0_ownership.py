# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0 ownership reconciliation, concern identity, and soundness gates."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.prod_q._p0_classification import classified_inventory
from tests.qualification.prod_q._p0_discovery import (
    class_method_self_calls,
    external_attribute_callers_of_method,
    module_contract_import_prefixes,
    modules_reachable_from_production_entrypoints,
    symbol_enclosing_class_by_module,
)
from tests.qualification.prod_q._p0_types import (
    AuxiliaryEvidenceStrength,
    AuxiliaryJustification,
    AuxiliaryJustificationKind,
    AuxiliaryReconciliationMetrics,
    AuxiliaryVerificationResult,
    ConcernFindingReconciliationMetrics,
    MechanismRole,
    OwnershipReconciliationReport,
    OwnershipResolutionState,
    ProdQFindingDisposition,
    ProductionReadinessCandidate,
    ProductionReadinessClassification,
    ProductionReadinessFamily,
    ProductionReadinessOwner,

)

_UNRESOLVED_CONTRACT_MARKERS = frozenset({"CONTRACT_UNRESOLVED", ""})
_UNRESOLVED_COMPOSITION_MARKERS = frozenset({"COMPOSITION_UNRESOLVED", ""})


def auxiliary_evidence_is_sufficient(record: ProductionReadinessClassification) -> bool:
    refs = record.current_head_evidence
    if any(ref.startswith("auxiliary_na:symbol:") for ref in refs):
        if not any(ref.startswith("auxiliary_evidence:") for ref in refs):
            return False
    if not any(ref.startswith("verification_result:verified") for ref in refs):
        return False
    if not any(ref.startswith("relationship_evidence:caller_to_callee:") for ref in refs):
        return False
    has_kind = any(ref.startswith("auxiliary_evidence:") for ref in refs)
    has_related = any(ref.startswith("related_concern:") for ref in refs)
    if not has_kind or not has_related:
        return False
    if any(
        ref == f"auxiliary_evidence:{AuxiliaryJustificationKind.MECHANICALLY_EXCLUDED_REACHABILITY.value}"
        for ref in refs
    ):
        return False
    return True


def is_auxiliary_discovery(record: ProductionReadinessClassification) -> bool:
    if record.mechanism_role != MechanismRole.AUXILIARY_DISCOVERY:
        return False
    return auxiliary_evidence_is_sufficient(record)


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


def _sanctioned_alternate_composition_keys() -> frozenset[str]:
    from tests.qualification.prod_q._p0_findings_register import known_prod_q_findings

    return frozenset(
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
        if len(owners) == 1 and len(contracts) == 1 and key in _sanctioned_alternate_composition_keys():
            continue
        violations.append(f"unsanctioned alternate composition for {key}")
    return tuple(violations)


def auxiliary_verified_without_relationship_proof_violations(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> tuple[str, ...]:
    inv = inventory if inventory is not None else classified_inventory()
    violations: list[str] = []
    for record in inv:
        if not is_auxiliary_discovery(record):
            continue
        refs = record.current_head_evidence
        if not any(ref.startswith("verification_result:verified") for ref in refs):
            continue
        if not any(ref.startswith("relationship_evidence:caller_to_callee:") for ref in refs):
            violations.append(
                f"{record.candidate_id}: private helper VERIFIED without caller_to_callee proof"
            )
        if not auxiliary_evidence_is_sufficient(record):
            violations.append(f"{record.candidate_id}: VERIFIED auxiliary fails sufficiency gate")
    return tuple(violations)


def auxiliary_na_without_symbol_evidence(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> tuple[str, ...]:
    inv = inventory if inventory is not None else classified_inventory()
    return tuple(
        f"{record.candidate_id}: auxiliary without evidenced justification"
        for record in inv
        if record.mechanism_role == MechanismRole.AUXILIARY_DISCOVERY
        and not auxiliary_evidence_is_sufficient(record)
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


def _pq_support_concern_key(family: ProductionReadinessFamily, indicator: str) -> str:
    return f"concern:pq_support:{family.value}:{indicator}"


def _contract_concern_for_import(prefix: str) -> str:
    return f"concern:import:{prefix}"


def _semantic_concerns_in_module(
    inventory: tuple[ProductionReadinessClassification, ...],
    module_path: str,
) -> tuple[ProductionReadinessClassification, ...]:
    return tuple(
        record
        for record in inventory
        if record.module == module_path and is_semantic_mechanism(record)
    )


def _auxiliary_source_symbol_identity(candidate: ProductionReadinessCandidate) -> str:
    return f"{candidate.module_path}::{candidate.symbol}"


def _operational_methods_in_class(
    module_path: str,
    class_name: str,
    module_semantics: tuple[ProductionReadinessClassification, ...],
    enclosing: dict[str, dict[str, str]],
    operational_indicators: frozenset[str],
) -> tuple[tuple[ProductionReadinessClassification, str], ...]:
    matched: list[tuple[ProductionReadinessClassification, str]] = []
    for semantic in module_semantics:
        semantic_indicator = semantic.responsibility.split(":", 1)[-1]
        if semantic_indicator not in operational_indicators:
            continue
        if "." in semantic.symbol:
            semantic_class, method = semantic.symbol.split(".", 1)
        else:
            semantic_class = enclosing.get(module_path, {}).get(semantic.symbol, "")
            method = semantic.symbol
        if semantic_class != class_name:
            continue
        matched.append((semantic, method))
    return tuple(matched)


def _verified_private_nested_helper(
    candidate: ProductionReadinessCandidate,
    module_semantics: tuple[ProductionReadinessClassification, ...],
    enclosing: dict[str, dict[str, str]],
    operational_indicators: frozenset[str],
) -> AuxiliaryJustification | None:
    if "." not in candidate.symbol:
        return None
    class_name, method_name = candidate.symbol.split(".", 1)
    if not method_name.startswith("_"):
        return None
    operational = _operational_methods_in_class(
        candidate.module_path,
        class_name,
        module_semantics,
        enclosing,
        operational_indicators,
    )
    if not operational:
        return None

    call_graph = class_method_self_calls(candidate.module_path)
    operational_callers: list[tuple[ProductionReadinessClassification, str]] = []
    for semantic, op_method in operational:
        invoked = call_graph.get((class_name, op_method), frozenset())
        if method_name in invoked:
            operational_callers.append((semantic, op_method))

    if not operational_callers:
        return None

    if external_attribute_callers_of_method(candidate.module_path, class_name, method_name):
        return None

    for caller_class, caller_method in call_graph:
        if caller_class != class_name:
            continue
        if method_name not in call_graph.get((caller_class, caller_method), frozenset()):
            continue
        if caller_method.startswith("_"):
            continue
        if any(op_method == caller_method for _, op_method in operational):
            continue
        return None

    concern_keys = {semantic.semantic_concern_key for semantic, _ in operational_callers}
    if len(concern_keys) != 1:
        return None

    semantic, op_method = operational_callers[0]
    source_location = f"{candidate.module_path}:{candidate.symbol}:L{candidate.line_no}"
    return AuxiliaryJustification(
        candidate_id=candidate.candidate_id,
        kind=AuxiliaryJustificationKind.HELPER_OF_CLASSIFIED_CONCERN,
        related_concern_key=semantic.semantic_concern_key,
        source_location=source_location,
        strength=AuxiliaryEvidenceStrength.PROVEN_STATIC,
        rationale="verified_private_nested_helper",
        source_symbol_identity=_auxiliary_source_symbol_identity(candidate),
        relationship_evidence=(
            f"caller_to_callee:{class_name}.{op_method}->{class_name}.{method_name}",
            f"ast_static_call:self.{method_name}",
            f"operational_semantic:{semantic.candidate_id}",
            f"analysis_limit:bounded_static_ast_no_dynamic_dispatch_proof",
            f"independent_authority:none_detected_in_module_static_shape",
        ),
        verification_result=AuxiliaryVerificationResult.VERIFIED,
    )


def _auxiliary_hint(
    candidate: ProductionReadinessCandidate,
    record: ProductionReadinessClassification,
    *,
    kind: AuxiliaryJustificationKind,
    related_concern_key: str,
    rationale: str,
    relationship_evidence: tuple[str, ...],
) -> AuxiliaryJustification:
    source_location = f"{candidate.module_path}:{candidate.symbol}:L{candidate.line_no}"
    return AuxiliaryJustification(
        candidate_id=record.candidate_id,
        kind=kind,
        related_concern_key=related_concern_key,
        source_location=source_location,
        strength=AuxiliaryEvidenceStrength.HINT_ONLY,
        rationale=rationale,
        source_symbol_identity=_auxiliary_source_symbol_identity(candidate),
        relationship_evidence=relationship_evidence,
        verification_result=AuxiliaryVerificationResult.HINT_ONLY,
    )


def build_auxiliary_justifications(
    raw: tuple[ProductionReadinessCandidate, ...],
    preliminary: tuple[ProductionReadinessClassification, ...],
) -> dict[str, AuxiliaryJustification]:
    from tests.qualification.prod_q._p0_classification import _OPERATIONAL_RESPONSIBILITY_INDICATORS

    raw_by_id = {candidate.candidate_id: candidate for candidate in raw}
    reachable = modules_reachable_from_production_entrypoints()
    enclosing = symbol_enclosing_class_by_module()
    contract_imports = module_contract_import_prefixes()
    semantics_by_module: dict[str, list[ProductionReadinessClassification]] = {}
    semantics_by_family: dict[ProductionReadinessFamily, list[ProductionReadinessClassification]] = {}
    for record in preliminary:
        if not is_semantic_mechanism(record):
            continue
        semantics_by_module.setdefault(record.module, []).append(record)
        semantics_by_family.setdefault(record.family, []).append(record)

    justifications: dict[str, AuxiliaryJustification] = {}
    for record in preliminary:
        if record.mechanism_role != MechanismRole.AUXILIARY_DISCOVERY:
            continue
        candidate = raw_by_id[record.candidate_id]
        module_semantics = tuple(semantics_by_module.get(candidate.module_path, []))
        verified = _verified_private_nested_helper(
            candidate,
            module_semantics,
            enclosing,
            _OPERATIONAL_RESPONSIBILITY_INDICATORS,
        )
        if verified is not None:
            justifications[record.candidate_id] = verified
            continue

        class_name = enclosing.get(candidate.module_path, {}).get(candidate.symbol, "")
        if class_name and module_semantics:
            for semantic in module_semantics:
                semantic_class = enclosing.get(candidate.module_path, {}).get(semantic.symbol, "")
                if semantic_class == class_name:
                    justifications[record.candidate_id] = _auxiliary_hint(
                        candidate,
                        record,
                        kind=AuxiliaryJustificationKind.HELPER_OF_CLASSIFIED_CONCERN,
                        related_concern_key=semantic.semantic_concern_key,
                        rationale="same_enclosing_class_hint_only",
                        relationship_evidence=(f"ast_enclosing_class:{class_name}",),
                    )
                    break
        if record.candidate_id in justifications:
            continue

        if module_semantics:
            same_family = [item for item in module_semantics if item.family == record.family]
            anchor = same_family[0] if same_family else module_semantics[0]
            justifications[record.candidate_id] = _auxiliary_hint(
                candidate,
                record,
                kind=AuxiliaryJustificationKind.HELPER_OF_CLASSIFIED_CONCERN,
                related_concern_key=anchor.semantic_concern_key,
                rationale="same_module_semantic_concern",
                relationship_evidence=(f"module:{candidate.module_path}",),
            )
            continue

        prefixes = contract_imports.get(candidate.module_path, ())
        if prefixes:
            justifications[record.candidate_id] = _auxiliary_hint(
                candidate,
                record,
                kind=AuxiliaryJustificationKind.IMPLEMENTATION_DETAIL_BEHIND_CONTRACT,
                related_concern_key=_contract_concern_for_import(prefixes[0]),
                rationale="module_imports_canonical_contract",
                relationship_evidence=(f"contract_import_module:{prefixes[0]}",),
            )
            continue

        if candidate.module_path not in reachable:
            justifications[record.candidate_id] = _auxiliary_hint(
                candidate,
                record,
                kind=AuxiliaryJustificationKind.MECHANICALLY_EXCLUDED_REACHABILITY,
                related_concern_key=_pq_support_concern_key(record.family, candidate.indicator),
                rationale="static_import_closure_unreachable_from_entrypoints",
                relationship_evidence=(
                    "reachability:bounded_static_import_closure",
                    "reachability:universal_exclusion_not_proven",
                ),
            )
            continue

        family_semantics = semantics_by_family.get(record.family, ())
        if family_semantics:
            justifications[record.candidate_id] = _auxiliary_hint(
                candidate,
                record,
                kind=AuxiliaryJustificationKind.NON_AUTHORITATIVE_PROJECTION,
                related_concern_key=family_semantics[0].semantic_concern_key,
                rationale="co_classified_family_semantic_anchor",
                relationship_evidence=(f"family:{record.family.value}",),
            )
            continue

        justifications[record.candidate_id] = _auxiliary_hint(
            candidate,
            record,
            kind=AuxiliaryJustificationKind.NON_AUTHORITATIVE_PROJECTION,
            related_concern_key=_pq_support_concern_key(record.family, candidate.indicator),
            rationale="pq_family_support_bucket",
            relationship_evidence=(f"family:{record.family.value}", f"indicator:{candidate.indicator}"),
        )
    return justifications


def apply_auxiliary_finalization(
    record: ProductionReadinessClassification,
    justification: AuxiliaryJustification | None,
) -> ProductionReadinessClassification:
    from dataclasses import replace

    from tests.qualification.prod_q._p0_types import ProductionReadinessEvidenceState

    if record.mechanism_role != MechanismRole.AUXILIARY_DISCOVERY:
        return record
    if justification is None:
        return _reclassify_unverified_auxiliary(record, ("auxiliary:missing_justification",))

    hint_refs = record.current_head_evidence + (
        f"source_symbol_identity:{justification.source_symbol_identity}",
        f"verification_result:{justification.verification_result.value}",
        f"auxiliary_rationale:{justification.rationale}",
    )
    for rel in justification.relationship_evidence:
        hint_refs = hint_refs + (f"relationship_evidence:{rel}",)

    if justification.verification_result != AuxiliaryVerificationResult.VERIFIED:
        return _reclassify_unverified_auxiliary(record, hint_refs)

    evidence_refs = hint_refs + (
        f"auxiliary_evidence:{justification.kind.value}",
        f"related_concern:{justification.related_concern_key}",
        f"source_location:{justification.source_location}",
        f"evidence_strength:{justification.strength.value}",
    )
    return replace(
        record,
        semantic_concern_key=f"auxiliary:{justification.related_concern_key}:{record.candidate_id}",
        ownership_resolution=OwnershipResolutionState.NOT_APPLICABLE_WITH_EVIDENCE,
        evidence_state=ProductionReadinessEvidenceState.N_A_WITH_EVIDENCE,
        current_head_evidence=evidence_refs,
    )


def _reclassify_unverified_auxiliary(
    record: ProductionReadinessClassification,
    hint_refs: tuple[str, ...],
) -> ProductionReadinessClassification:
    from dataclasses import replace

    from tests.qualification.prod_q._p0_types import ProductionReadinessEvidenceState

    indicator = record.responsibility.split(":", 1)[-1]
    concern_key = _pq_support_concern_key(record.family, indicator)
    return replace(
        record,
        mechanism_role=MechanismRole.SEMANTIC_MECHANISM,
        semantic_concern_key=concern_key,
        ownership_resolution=OwnershipResolutionState.NOT_YET_RESOLVED,
        evidence_state=ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        current_head_evidence=hint_refs + ("auxiliary_reclassified:unverified_relationship",),
    )


def build_auxiliary_reconciliation_metrics(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> AuxiliaryReconciliationMetrics:
    inv = inventory if inventory is not None else classified_inventory()
    discovered = sum(1 for record in inv if record.mechanism_role == MechanismRole.AUXILIARY_DISCOVERY)
    justified = sum(1 for record in inv if is_auxiliary_discovery(record))
    unresolved = discovered - justified
    falsely_excluded = sum(
        1
        for record in inv
        if record.mechanism_role == MechanismRole.AUXILIARY_DISCOVERY
        and any(ref.startswith("auxiliary_na:symbol:") for ref in record.current_head_evidence)
    )
    return AuxiliaryReconciliationMetrics(
        auxiliary_discovered=discovered,
        auxiliary_justified=justified,
        auxiliary_unresolved=unresolved,
        auxiliary_falsely_excluded=falsely_excluded,
    )


def probe_concern_identity_stable_under_candidate_order() -> str | None:
    from tests.qualification.prod_q._p0_classification import classify_candidate
    from tests.qualification.prod_q._p0_discovery import raw_discovery_candidates

    raw = raw_discovery_candidates()
    fwd = {c.candidate_id: classify_candidate(c).semantic_concern_key for c in raw}
    rev = {c.candidate_id: classify_candidate(c).semantic_concern_key for c in reversed(raw)}
    if fwd != rev:
        return "semantic_concern_key depends on discovery iteration order"
    return None
