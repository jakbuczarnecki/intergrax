# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0 inventory identity, evidence, and mutation integrity gates."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from tests.qualification.prod_q._p0_classification import classify_candidate
from tests.qualification.prod_q._p0_discovery import raw_discovery_candidates
from tests.qualification.prod_q._p0_types import (
    CandidateInventoryStatus,
    DiscoveryKind,
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
        )
    ):
        return "DISCOVERED_SURFACE without DISCOVERED/NOT_YET_RESOLVED state"
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
