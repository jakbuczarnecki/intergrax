# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0 inventory aggregation, family coverage, ownership matrix."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.prod_q._p0_classification import (
    _ANCHOR_METADATA,
    classification_by_candidate_id,
    classified_inventory,
)
from tests.qualification.prod_q._p0_discovery import scanned_module_paths
from tests.qualification.prod_q._p0_discovery import (
    anchor_resolution_failures,
    discovery_scan_failures,
    production_files_scanned_count,
    raw_discovery_candidates,
    raw_discovery_duplicate_ids,
)
from tests.qualification.prod_q._p0_integrity import unclassified_indicator_ids
from tests.qualification.prod_q._p0_types import (
    ALL_PRODUCTION_READINESS_FAMILIES,
    ClosedWorldDiscoveryReport,
    DiscoveryKind,
    FamilyCoverageRecord,
    OwnershipMatrixReport,
    OwnershipRow,
    OwnershipResolutionState,
    ProdQP0Matrix,
    ProdQFindingDisposition,
    ProductionReadinessEvidenceState,
    ProductionReadinessFamily,
    ProductionReadinessOwner,
)

_FAMILY_TITLES: dict[ProductionReadinessFamily, str] = {
    ProductionReadinessFamily.PQ_F01: "production mode / environment semantics",
    ProductionReadinessFamily.PQ_F02: "startup/bootstrap",
    ProductionReadinessFamily.PQ_F03: "shutdown/drain/resource cleanup",
    ProductionReadinessFamily.PQ_F04: "health",
    ProductionReadinessFamily.PQ_F05: "readiness",
    ProductionReadinessFamily.PQ_F06: "degraded-state semantics",
    ProductionReadinessFamily.PQ_F07: "terminal/critical failure visibility",
    ProductionReadinessFamily.PQ_F08: "operator-actionable recovery semantics",
    ProductionReadinessFamily.PQ_F09: "provider/plugin production admission and trust",
    ProductionReadinessFamily.PQ_F10: "unsupported production configuration",
    ProductionReadinessFamily.PQ_F11: "production bypass / strict-vs-lab separation",
    ProductionReadinessFamily.PQ_F12: "secrets / sensitive-provider boundary",
    ProductionReadinessFamily.PQ_F13: "capacity / root work admission",
    ProductionReadinessFamily.PQ_F14: "queue / backpressure / overload",
    ProductionReadinessFamily.PQ_F15: "provider throttling / rate limit",
    ProductionReadinessFamily.PQ_F16: "timeout / retry / cancellation ownership",
    ProductionReadinessFamily.PQ_F17: "idempotency / external-operation termination",
    ProductionReadinessFamily.PQ_F18: "production dependency materialization",
    ProductionReadinessFamily.PQ_F19: "sandbox/isolation production guarantees",
    ProductionReadinessFamily.PQ_F20: "retention/deletion/redaction responsibility",
    ProductionReadinessFamily.PQ_F21: "platform-vs-infrastructure responsibility",
    ProductionReadinessFamily.PQ_F22: "tenant continuity on production-specific paths",
    ProductionReadinessFamily.PQ_F23: "backup/recovery operational responsibility",
}


@lru_cache(maxsize=1)
def build_closed_world_discovery_report() -> ClosedWorldDiscoveryReport:
    raw = raw_discovery_candidates()
    by_id = classification_by_candidate_id()
    unclassified = unclassified_indicator_ids(by_id)
    return ClosedWorldDiscoveryReport(
        module_paths_scanned=scanned_module_paths(),
        scan_failures=discovery_scan_failures(),
        anchor_resolution_failures=anchor_resolution_failures(),
        raw_candidates=raw,
        duplicate_candidate_ids=raw_discovery_duplicate_ids(),
        unclassified_candidate_ids=unclassified,
    )


def _family_evidence_state(
    family: ProductionReadinessFamily,
    records: tuple,
) -> ProductionReadinessEvidenceState:
    states = {r.evidence_state for r in records}
    if ProductionReadinessEvidenceState.BLOCKER in states:
        return ProductionReadinessEvidenceState.BLOCKER
    if ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED in states:
        return ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED
    if ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD in states:
        return ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD
    if ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED in states:
        return ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED
    return ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD


@lru_cache(maxsize=1)
def build_family_coverage() -> tuple[FamilyCoverageRecord, ...]:
    inventory = classified_inventory()
    out: list[FamilyCoverageRecord] = []
    for family in sorted(ALL_PRODUCTION_READINESS_FAMILIES, key=lambda f: f.value):
        members = tuple(r for r in inventory if r.family == family)
        evidence = _family_evidence_state(family, members)
        blockers = [r for r in members if r.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER]
        disposition = (
            ProdQFindingDisposition.IN_SCOPE_BLOCKER
            if blockers
            else ProdQFindingDisposition.TRACKED_FREEZE_DEBT
        )
        if family == ProductionReadinessFamily.PQ_F21:
            evidence = ProductionReadinessEvidenceState.N_A_WITH_EVIDENCE
            disposition = ProdQFindingDisposition.TRACKED_FREEZE_DEBT
        out.append(
            FamilyCoverageRecord(
                family=family,
                title=_FAMILY_TITLES[family],
                evidence_state=evidence,
                finding_disposition=disposition,
                candidate_count=len(members),
                evidence_refs=tuple(
                    sorted({ref for r in members for ref in r.current_head_evidence})[:8]
                ),
            )
        )
    return tuple(out)


@lru_cache(maxsize=1)
def build_ownership_matrix() -> OwnershipMatrixReport:
    inventory = classified_inventory()
    raw_by_id = {c.candidate_id: c for c in raw_discovery_candidates()}
    by_key: dict[str, list[tuple[str, ProductionReadinessOwner, str, str, OwnershipResolutionState]]] = {}
    for record in inventory:
        key = record.semantic_responsibility_key
        by_key.setdefault(key, []).append(
            (
                record.candidate_id,
                record.semantic_owner,
                record.canonical_contract,
                record.composition_owner,
                record.ownership_resolution,
            )
        )

    rows: list[OwnershipRow] = []
    duplicate_semantic: list[str] = []
    duplicate_composition: list[str] = []
    alternate_contracts: list[str] = []
    unresolved: list[str] = []
    concern_contracts: dict[str, set[str]] = {}

    for key, items in sorted(by_key.items()):
        owners = {item[1] for item in items}
        compositions = {item[3] for item in items}
        contracts = {item[2] for item in items}
        resolutions = {item[4] for item in items}
        cid_tuple = tuple(item[0] for item in items)
        if len(owners) > 1:
            duplicate_semantic.append(key)
        if len(compositions) > 1 and ProductionReadinessOwner.DISCOVERED_SURFACE not in owners:
            duplicate_composition.append(key)
        primary_owner = next(iter(sorted(owners, key=lambda o: o.value)))
        sample_cid = cid_tuple[0]
        raw_candidate = raw_by_id.get(sample_cid)
        if raw_candidate and raw_candidate.discovery_kind == DiscoveryKind.ANCHOR_REGISTRY:
            anchor_key = (raw_candidate.module_path, raw_candidate.symbol)
            if anchor_key in _ANCHOR_METADATA:
                if OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE not in resolutions:
                    unresolved.append(key)
        rows.append(
            OwnershipRow(
                responsibility_key=key,
                semantic_owner=primary_owner,
                canonical_contract=next(iter(contracts)),
                composition_owner=next(iter(compositions)),
                candidate_ids=cid_tuple,
            )
        )

    for record in inventory:
        if record.ownership_resolution != OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE:
            continue
        if record.canonical_contract.startswith("CONTRACT_"):
            continue
        concern_key = record.responsibility
        concern_contracts.setdefault(concern_key, set()).add(record.canonical_contract)
    for concern_key, contracts in concern_contracts.items():
        if len(contracts) > 1:
            alternate_contracts.append(concern_key)

    return OwnershipMatrixReport(
        rows=tuple(rows),
        duplicate_semantic_owner_keys=tuple(sorted(frozenset(duplicate_semantic))),
        duplicate_composition_owner_keys=tuple(sorted(frozenset(duplicate_composition))),
        alternate_canonical_contract_concern_keys=tuple(sorted(alternate_contracts)),
        unresolved_ownership_keys=tuple(sorted(frozenset(unresolved))),
    )


def build_p0_matrix(frz_count: int) -> ProdQP0Matrix:
    report = build_closed_world_discovery_report()
    blockers = sum(
        1 for r in classified_inventory() if r.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER
    )
    return ProdQP0Matrix(
        production_files_scanned=production_files_scanned_count(),
        candidates_discovered=len(report.raw_candidates),
        candidates_classified=len(classified_inventory()),
        unclassified_candidates=len(report.unclassified_candidate_ids),
        families_covered=len(ALL_PRODUCTION_READINESS_FAMILIES),
        frz_criteria_classified=frz_count,
        in_scope_blockers=blockers,
    )


def get_classified_inventory() -> tuple:
    return classified_inventory()
