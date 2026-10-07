# © Artur Czarnecki. All rights reserved.

"""STATE-X-FINAL-R1 closed-world durable mechanism discovery and classification SSOT."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.state_x._state_x_explicit_mechanism_classifications import (
    EXPLICIT_MECHANISM_CLASSIFICATIONS,
    ExplicitMechanismClassification,
)
from tests.qualification.state_x.inventory import (
    CURRENT_STATE_X_FAMILY_IDS,
    HISTORICAL_BASE_FAMILY_IDS,
    ProjectionOrTruth,
    SemanticOwnershipRole,
    STATE_X_FAMILY_INVENTORY,
    StateFamilyInventoryEntry,
)

_FORBIDDEN_PLACEHOLDER_OWNERS: Final[frozenset[str]] = frozenset(
    {
        "see family contract",
        "see classes in module",
        "module aggregate",
        "non execution-state subsystem",
        "unmapped durable-like symbol",
    },
)

_FORBIDDEN_OWNER_STAGES: Final[frozenset[str]] = frozenset(
    {
        "",
        "REVIEW-QUEUE",
        "UNKNOWN",
        "PENDING",
        "TBD",
        "MISC",
        "OTHER",
        "UNREVIEWED",
        "AUTO_CLASSIFIED",
    },
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

DISCOVERY_SCAN_ROOTS: Final[tuple[str, ...]] = (
    "intergrax/contracts",
    "intergrax/runtime",
    "agents",
    "applications",
    "intergrax/applications",
)

_DISCOVERY_EXCLUDE_PREFIXES: Final[tuple[str, ...]] = (
    "applications/lab_application/docker/runtime-context/",
    "applications/local_workspace_application/docker/runtime-context/",
)

_CLASS_SUFFIX = re.compile(
    r"(Persistence|StateStore|CorrelationStore|AuthorityStore|LifecycleStore|"
    r"DurableBacking|DurableState|CheckpointStore|CheckpointPersistence|"
    r"QueueStore|IdempotencyStore|CompensationQueueStore|EventStore|RunTraceStore|"
    r"Ledger|Repository|BackingStore)$",
)
_FILE_PATTERN = re.compile(r"(store|persistence|ledger|checkpoint_store)\.py$", re.IGNORECASE)

PRIOR_BROAD_EXCLUSION_PREFIXES: Final[tuple[str, ...]] = (
    "intergrax/runtime/adaptive/",
    "intergrax/runtime/prediction/",
    "intergrax/runtime/self_healing/",
    "intergrax/runtime/vendor_knowledge/",
    "intergrax/runtime/diagnostics/",
    "intergrax/runtime/governance/",
    "intergrax/runtime/observability/",
    "intergrax/runtime/notifications/",
    "intergrax/runtime/organization/",
    "intergrax/runtime/integrations/",
    "intergrax/runtime/external_operations/",
    "intergrax/runtime/execution_evidence/",
    "intergrax/runtime/background_execution/",
    "intergrax/runtime/nexus/",
    "intergrax/runtime/execution/continuation/",
    "intergrax/runtime/execution/deadline_authority/",
    "intergrax/runtime/execution/delegated_execution/",
    "intergrax/runtime/execution/suspended_operation/",
    "intergrax/runtime/execution/active_",
    "intergrax/runtime/execution/decision_finalization/",
    "intergrax/runtime/task_memory/",
)


class DurableStateClassification(StrEnum):
    CANONICAL_FAMILY = "canonical_family"
    NEW_CANONICAL_FAMILY = "new_canonical_family"
    COMPONENT_OF_FAMILY = "component_of_family"
    PROVIDER_IMPLEMENTATION = "provider_implementation"
    NON_AUTHORITATIVE_PROJECTION = "non_authoritative_projection"
    OUTSIDE_STATE_X = "outside_state_x"
    NON_DURABLE_REFERENCE_ONLY = "non_durable_reference_only"


@dataclass(frozen=True, slots=True)
class DurableStateMechanismRecord:
    mechanism_id: str
    symbol: str
    paths: tuple[str, ...]
    classification: DurableStateClassification
    family_id: str | None
    semantic_owner: str | None
    composition_owner: str | None
    durability: str
    canonical_truth: bool
    tenant_semantics: str
    identity_semantics: str
    authority_semantics: str
    atomicity_semantics: str
    stale_conflict_behavior: str
    corruption_behavior: str
    restart_restore_behavior: str
    backup_restore_responsibility: str
    owner_stage: str
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscoveredMechanism:
    mechanism_id: str
    symbol: str
    paths: tuple[str, ...]
    discovery_kind: str


def _inventory_symbol_family_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in STATE_X_FAMILY_INVENTORY:
        for sym in entry.implementation_symbols:
            mapping[sym] = entry.family_id
        for cref in entry.contract_references:
            mapping[cref.symbol] = entry.family_id
        mapping[entry.semantic_owner] = entry.family_id
    return mapping


_SYMBOL_FAMILY: Final[dict[str, str]] = _inventory_symbol_family_map()

_CANONICAL_CONTRACT_SYMBOLS: Final[frozenset[str]] = frozenset(
    cref.symbol
    for entry in STATE_X_FAMILY_INVENTORY
    for cref in entry.contract_references
)

_FAMILY_BY_ID: Final[dict[str, StateFamilyInventoryEntry]] = {
    e.family_id: e for e in STATE_X_FAMILY_INVENTORY
}


def _entry(family_id: str) -> StateFamilyInventoryEntry:
    return _FAMILY_BY_ID[family_id]


def _excluded(rel: str) -> bool:
    if "test_" in rel or "/tests/" in rel:
        return True
    return any(rel.startswith(p) for p in _DISCOVERY_EXCLUDE_PREFIXES)


def discover_durable_state_candidates() -> tuple[DiscoveredMechanism, ...]:
    found: dict[str, DiscoveredMechanism] = {}
    for root in DISCOVERY_SCAN_ROOTS:
        base = _REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if _excluded(rel):
                continue
            if _FILE_PATTERN.search(path.name):
                mid = f"path:{rel}"
                found[mid] = DiscoveredMechanism(
                    mechanism_id=mid,
                    symbol=path.stem,
                    paths=(rel,),
                    discovery_kind="file",
                )
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if node.name.endswith("Registry"):
                    continue
                if not _CLASS_SUFFIX.search(node.name):
                    continue
                mid = f"class:{rel}::{node.name}"
                found[mid] = DiscoveredMechanism(
                    mechanism_id=mid,
                    symbol=node.name,
                    paths=(rel,),
                    discovery_kind="class",
                )
    return tuple(sorted(found.values(), key=lambda c: c.mechanism_id))


def _base_record_fields(
    candidate: DiscoveredMechanism,
    classification: DurableStateClassification,
    family_id: str | None,
    *,
    semantic_owner: str | None = None,
    composition_owner: str | None = None,
    durability: str = "provider-qualified",
    canonical_truth: bool = False,
    tenant_semantics: str = "see family contract",
    identity_semantics: str = "see family contract",
    authority_semantics: str = "does not mint governance permission",
    atomicity_semantics: str = "provider CAS where applicable",
    stale_conflict_behavior: str = "fail closed on conflict",
    corruption_behavior: str = "explicit rejection on corrupt bytes",
    restart_restore_behavior: str = "reload durable record on restart",
    backup_restore_responsibility: str = "BACKEND/OPERATOR WITH PLATFORM CONSISTENCY",
    owner_stage: str = "STATE-X",
    evidence: tuple[str, ...] = (),
) -> DurableStateMechanismRecord:
    if family_id and semantic_owner is None:
        semantic_owner = _entry(family_id).semantic_owner
    if family_id and composition_owner is None:
        composition_owner = _entry(family_id).composition_owner
    return DurableStateMechanismRecord(
        mechanism_id=candidate.mechanism_id,
        symbol=candidate.symbol,
        paths=candidate.paths,
        classification=classification,
        family_id=family_id,
        semantic_owner=semantic_owner,
        composition_owner=composition_owner,
        durability=durability,
        canonical_truth=canonical_truth,
        tenant_semantics=tenant_semantics,
        identity_semantics=identity_semantics,
        authority_semantics=authority_semantics,
        atomicity_semantics=atomicity_semantics,
        stale_conflict_behavior=stale_conflict_behavior,
        corruption_behavior=corruption_behavior,
        restart_restore_behavior=restart_restore_behavior,
        backup_restore_responsibility=backup_restore_responsibility,
        owner_stage=owner_stage,
        evidence=evidence,
    )


def _outside(
    candidate: DiscoveredMechanism,
    owner_stage: str,
    reason: str,
    *,
    durability: str = "N/A",
    evidence: tuple[str, ...] | None = None,
) -> DurableStateMechanismRecord:
    return _base_record_fields(
        candidate,
        DurableStateClassification.OUTSIDE_STATE_X,
        None,
        semantic_owner=reason,
        composition_owner=reason,
        durability=durability,
        canonical_truth=False,
        tenant_semantics="N/A",
        identity_semantics="N/A",
        authority_semantics="cannot become execution/recovery truth",
        atomicity_semantics="N/A",
        stale_conflict_behavior="N/A",
        corruption_behavior="N/A",
        restart_restore_behavior="N/A",
        backup_restore_responsibility="N/A",
        owner_stage=owner_stage,
        evidence=evidence if evidence is not None else (reason,),
    )


def _projection(candidate: DiscoveredMechanism, reason: str) -> DurableStateMechanismRecord:
    return _base_record_fields(
        candidate,
        DurableStateClassification.NON_AUTHORITATIVE_PROJECTION,
        None,
        semantic_owner=reason,
        composition_owner="observability composition",
        durability="durable optional; non-authoritative",
        canonical_truth=False,
        authority_semantics="observes; does not constrain execution authority alone",
        owner_stage="TRACE-X",
        evidence=(reason,),
    )


def _reference_only(candidate: DiscoveredMechanism, owner_family: str | None) -> DurableStateMechanismRecord:
    return _base_record_fields(
        candidate,
        DurableStateClassification.NON_DURABLE_REFERENCE_ONLY,
        owner_family,
        durability="in-process / test reference",
        canonical_truth=False,
        restart_restore_behavior="not production durable truth",
        backup_restore_responsibility="NOT_DURABLE_REFERENCE_ONLY",
        owner_stage="STATE-X",
        evidence=("InMemory or single-process reference provider",),
    )


def _record_from_explicit(
    candidate: DiscoveredMechanism,
    spec: ExplicitMechanismClassification,
) -> DurableStateMechanismRecord:
    classification = DurableStateClassification(spec.classification)
    return _base_record_fields(
        candidate,
        classification,
        spec.family_id,
        semantic_owner=spec.semantic_owner,
        composition_owner=spec.composition_owner,
        durability=spec.durability,
        canonical_truth=spec.canonical_truth,
        tenant_semantics=spec.tenant_semantics,
        identity_semantics=spec.identity_semantics,
        authority_semantics=spec.authority_semantics,
        atomicity_semantics=spec.atomicity_semantics,
        stale_conflict_behavior=spec.stale_conflict_behavior,
        corruption_behavior=spec.corruption_behavior,
        restart_restore_behavior=spec.restart_restore_behavior,
        backup_restore_responsibility=spec.backup_restore_responsibility,
        owner_stage=spec.owner_stage,
        evidence=spec.evidence,
    )


def classify_candidate(
    candidate: DiscoveredMechanism,
) -> DurableStateMechanismRecord | None:
    """Return explicit classification or None when the candidate is unclassified."""
    explicit = EXPLICIT_MECHANISM_CLASSIFICATIONS.get(candidate.mechanism_id)
    if explicit is not None:
        return _record_from_explicit(candidate, explicit)

    sym = candidate.symbol
    rel = candidate.paths[0]

    if rel.startswith("intergrax/runtime/task_memory/") or sym in (
        "TaskMemoryPersistence",
        "NullTaskMemoryPersistence",
    ):
        return _outside(
            candidate,
            "APPLICATION-MEMORY",
            "task memory plane; agent/application scoped memory not platform execution SSOT",
            evidence=(
                "intergrax/runtime/task_memory/ — application memory plane",
                "tests/qualification/state_x/_state_x_final_r1_closed_world_tests.py::test_sxf_r1_q29",
            ),
        )

    if sym in _CANONICAL_CONTRACT_SYMBOLS and sym in _SYMBOL_FAMILY:
        fid = _SYMBOL_FAMILY[sym]
        is_new = fid not in HISTORICAL_BASE_FAMILY_IDS
        inv = _entry(fid)
        return _base_record_fields(
            candidate,
            DurableStateClassification.NEW_CANONICAL_FAMILY
            if is_new
            else DurableStateClassification.CANONICAL_FAMILY,
            fid,
            semantic_owner=inv.semantic_owner,
            composition_owner=inv.composition_owner,
            canonical_truth=True,
            evidence=(f"canonical contract {sym}", inv.contract_references[0].path),
        )

    if sym in _SYMBOL_FAMILY:
        fid = _SYMBOL_FAMILY[sym]
        inv = _entry(fid)
        if sym == inv.semantic_owner or sym in {c.symbol for c in inv.contract_references}:
            cls = (
                DurableStateClassification.NEW_CANONICAL_FAMILY
                if fid not in HISTORICAL_BASE_FAMILY_IDS
                else DurableStateClassification.CANONICAL_FAMILY
            )
            return _base_record_fields(
                candidate,
                cls,
                fid,
                semantic_owner=inv.semantic_owner,
                composition_owner=inv.composition_owner,
                canonical_truth=inv.projection_or_truth.value.startswith("CANONICAL"),
                evidence=(f"inventory semantic owner/provider {sym}",),
            )
        if sym.endswith("BackingStore") or sym.endswith("Backing"):
            return _base_record_fields(
                candidate,
                DurableStateClassification.COMPONENT_OF_FAMILY,
                fid,
                semantic_owner=inv.semantic_owner,
                composition_owner=inv.composition_owner,
                canonical_truth=True,
                evidence=("physical backing component of family store",),
            )
        if sym.startswith("InMemory"):
            return _reference_only(candidate, fid)
        return _base_record_fields(
            candidate,
            DurableStateClassification.PROVIDER_IMPLEMENTATION,
            fid,
            semantic_owner=inv.semantic_owner,
            composition_owner=inv.composition_owner,
            evidence=(f"provider implementation for {fid}",),
        )

    if rel.startswith("applications/governed_contractor_application/") and "ContinuationStateStore" in sym:
        inv = _entry("SX-F17")
        return _base_record_fields(
            candidate,
            DurableStateClassification.COMPONENT_OF_FAMILY,
            "SX-F17",
            semantic_owner=inv.semantic_owner,
            composition_owner=inv.composition_owner,
            evidence=("application host adapter over execution continuation contract",),
        )

    return None


@lru_cache(maxsize=1)
def _build_closed_world_inventory() -> tuple[DurableStateMechanismRecord, ...]:
    records: list[DurableStateMechanismRecord] = []
    unclassified: list[str] = []
    for candidate in discover_durable_state_candidates():
        record = classify_candidate(candidate)
        if record is None:
            unclassified.append(candidate.mechanism_id)
        else:
            records.append(record)
    assert not unclassified, f"unclassified durable mechanisms: {unclassified[:20]}"
    by_id = {r.mechanism_id: r for r in records}
    assert len(by_id) == len(records), "duplicate mechanism_id in classification"
    return tuple(records)


STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY: Final[tuple[DurableStateMechanismRecord, ...]] = (
    _build_closed_world_inventory()
)


def _record_has_placeholder_owner(record: DurableStateMechanismRecord) -> bool:
    hay = " ".join(
        (
            record.semantic_owner or "",
            record.composition_owner or "",
            " ".join(record.evidence),
        ),
    ).lower()
    return any(p in hay for p in _FORBIDDEN_PLACEHOLDER_OWNERS)


def assert_no_review_queue_records() -> None:
    hits = [r for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY if r.owner_stage in _FORBIDDEN_OWNER_STAGES]
    assert not hits, f"forbidden owner_stage records: {[r.mechanism_id for r in hits[:10]]}"


def assert_no_anonymous_components() -> None:
    hits = [
        r
        for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
        if r.classification == DurableStateClassification.COMPONENT_OF_FAMILY and r.family_id is None
    ]
    assert not hits, f"anonymous components: {[r.mechanism_id for r in hits[:10]]}"


def assert_no_anonymous_providers() -> None:
    hits = [
        r
        for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
        if r.classification == DurableStateClassification.PROVIDER_IMPLEMENTATION and r.family_id is None
    ]
    assert not hits, f"anonymous providers: {[r.mechanism_id for r in hits[:10]]}"


def assert_no_anonymous_reference_implementations() -> None:
    hits = [
        r
        for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY
        if r.classification == DurableStateClassification.NON_DURABLE_REFERENCE_ONLY
        and r.family_id is None
    ]
    assert not hits, f"anonymous reference stores: {[r.mechanism_id for r in hits[:10]]}"


def assert_no_placeholder_classification_records() -> None:
    hits = [r for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY if _record_has_placeholder_owner(r)]
    assert not hits, f"placeholder classification records: {[r.mechanism_id for r in hits[:10]]}"


def assert_unknown_candidate_is_rejected() -> None:
    unknown = DiscoveredMechanism(
        mechanism_id="class:intergrax/runtime/synthetic.py::BrandNewDurableFooStateStore",
        symbol="BrandNewDurableFooStateStore",
        paths=("intergrax/runtime/synthetic.py",),
        discovery_kind="class",
    )
    assert classify_candidate(unknown) is None


def assert_all_discovered_candidates_explicitly_classified() -> None:
    discovered = {c.mechanism_id for c in discover_durable_state_candidates()}
    classified = {r.mechanism_id for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY}
    unknown = discovered - classified
    assert not unknown, f"unclassified discovered={sorted(unknown)[:20]}"
    assert not (classified - discovered), "orphan classification records"


def assert_durable_state_discovery_fully_classified() -> None:
    assert_all_discovered_candidates_explicitly_classified()
    assert_no_review_queue_records()
    assert_no_anonymous_components()
    assert_no_anonymous_providers()
    assert_no_anonymous_reference_implementations()
    assert_no_placeholder_classification_records()
    for record in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY:
        assert record.classification in DurableStateClassification
        forbidden = {"UNKNOWN", "TBD", "IGNORE", "MISC", "OTHER", "ALLOWLIST"}
        blob = record.classification.value.upper()
        assert not any(tok in blob for tok in forbidden)
        if record.classification in (
            DurableStateClassification.CANONICAL_FAMILY,
            DurableStateClassification.NEW_CANONICAL_FAMILY,
        ):
            assert record.family_id in CURRENT_STATE_X_FAMILY_IDS
            assert record.semantic_owner
            assert record.composition_owner
            assert record.canonical_truth is True
        if record.classification == DurableStateClassification.OUTSIDE_STATE_X:
            assert record.owner_stage not in _FORBIDDEN_OWNER_STAGES
            assert record.semantic_owner
            assert record.evidence
            assert record.family_id is None
            assert record.canonical_truth is False
        if record.classification == DurableStateClassification.PROVIDER_IMPLEMENTATION:
            assert record.family_id in CURRENT_STATE_X_FAMILY_IDS
        if record.classification == DurableStateClassification.COMPONENT_OF_FAMILY:
            assert record.family_id is not None
        if record.classification == DurableStateClassification.NON_AUTHORITATIVE_PROJECTION:
            assert record.canonical_truth is False
            assert record.evidence
            assert record.owner_stage not in _FORBIDDEN_OWNER_STAGES
        if record.classification == DurableStateClassification.NON_DURABLE_REFERENCE_ONLY:
            assert record.family_id is not None


def assert_no_blind_directory_exclusions_in_scanner() -> None:
    text = Path(__file__).with_name("_state_x_final_support.py").read_text(encoding="utf-8")
    assert "out_of_state_x_prefixes" not in text


def assert_current_family_registry_complete() -> None:
    assert HISTORICAL_BASE_FAMILY_IDS == tuple(f"SX-F{i:02d}" for i in range(1, 16))
    assert set(HISTORICAL_BASE_FAMILY_IDS) <= set(CURRENT_STATE_X_FAMILY_IDS)
    owners: dict[str, str] = {}
    for entry in STATE_X_FAMILY_INVENTORY:
        if entry.semantic_ownership_role in (
            SemanticOwnershipRole.CANONICAL_OWNER,
            SemanticOwnershipRole.COMPONENT_OWNER,
        ):
            if entry.projection_or_truth in (
                ProjectionOrTruth.CANONICAL_TRUTH,
                ProjectionOrTruth.DURABLE_COMPONENT_OF_CANONICAL_TRUTH,
            ):
                key = entry.semantic_owner
                assert key not in owners or owners[key] == entry.family_id, (
                    f"duplicate semantic owner {key}"
                )
                owners[key] = entry.family_id


def scan_unclassified_durable_persistence_paths() -> tuple[str, ...]:
    """Legacy filename scan — must remain empty when classification SSOT is complete."""
    hits: list[str] = []
    for root in ("intergrax/runtime/", "agents/", "applications/", "intergrax/applications/"):
        base = _REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if _excluded(rel):
                continue
            if not _FILE_PATTERN.search(path.name):
                continue
            mid = f"path:{rel}"
            record = next(
                (r for r in STATE_X_DURABLE_STATE_CLOSED_WORLD_INVENTORY if r.mechanism_id == mid),
                None,
            )
            if record is None:
                hits.append(rel)
    return tuple(sorted(hits))
