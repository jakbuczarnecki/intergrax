# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-R2 qualification gates."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tests.qualification.compat_x._compat_x_legacy_obligations import (
    frz_cmp_06_family_ids,
    legacy_persisted_compatibility_obligations,
)
from tests.qualification.compat_x._compat_x_persisted_inventory import PERSISTED_SCHEMA_FAMILIES
from tests.qualification.compat_x._compat_x_persisted_types import (
    LegacyCompatibilityDisposition,
    MigrationDisposition,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_FORBIDDEN_IMPLICIT_PERSISTED_PATHS = (
    "intergrax/contracts/runtime_event.py",
    "intergrax/contracts/execution/suspended_operation/descriptor.py",
    "intergrax/agent_distribution/sqlite_payload_legacy.py",
    "intergrax/memory/user_profile_serialization.py",
    "intergrax/memory/conversational_memory_persistence.py",
    "intergrax/applications/_shared/registry_snapshot_persistence.py",
)


@dataclass(frozen=True, slots=True)
class PersistedR2R2Metrics:
    legacy_compatibility_obligations: int
    explicit_migrations: int
    explicit_legacy_readers: int
    current_only_fail_closed: int
    no_prior_version: int
    deferred_event_evolution: int
    unclassified_legacy_obligations: int
    migration_owner_missing: int
    duplicate_migration_authority: int
    implicit_compatibility_branches: int


def _family_by_id(family_id: str):
    return next(f for f in PERSISTED_SCHEMA_FAMILIES if f.family_id == family_id)


def _duplicate_migration_owners() -> tuple[str, ...]:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for obligation in legacy_persisted_compatibility_obligations():
        owner = obligation.migration_owner_path
        if owner is None:
            continue
        prior = seen.get(obligation.family_id)
        if prior is not None and prior != owner:
            duplicates.append(obligation.family_id)
        seen.setdefault(obligation.family_id, owner)
    return tuple(sorted(set(duplicates)))


def _implicit_before_validators(module_rel: str) -> list[str]:
    path = _REPO_ROOT / module_rel.replace("/", "\\") if False else _REPO_ROOT / module_rel
    if not path.is_file():
        return [f"missing module {module_rel}"]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_migrate_legacy"):
            hits.append(f"{module_rel}:{node.lineno}:{node.name}")
    return hits


@lru_cache(maxsize=1)
def compute_persisted_r2_r2_metrics() -> PersistedR2R2Metrics:
    obligations = legacy_persisted_compatibility_obligations()
    by_family: dict[str, LegacyCompatibilityDisposition] = {}
    unclassified = 0
    migration_owner_missing = 0
    for item in obligations:
        if item.disposition == LegacyCompatibilityDisposition.DEFERRED_TO_EVENT_EVOLUTION:
            continue
        by_family[item.family_id] = item.disposition
        if item.disposition in (
            LegacyCompatibilityDisposition.EXPLICIT_MIGRATION,
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
        ) and item.migration_owner_path is None:
            migration_owner_missing += 1

    covered = frozenset(by_family)
    for family_id in frz_cmp_06_family_ids():
        if family_id not in covered:
            unclassified += 1

    implicit = 0
    for rel in (
        "intergrax/contracts/execution/suspended_operation/descriptor.py",
    ):
        implicit += len(_implicit_before_validators(rel))

    explicit_migrations = sum(
        1 for o in obligations if o.disposition == LegacyCompatibilityDisposition.EXPLICIT_MIGRATION
    )
    explicit_readers = sum(
        1
        for o in obligations
        if o.disposition == LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER
    )
    current_only = sum(
        1
        for o in obligations
        if o.disposition == LegacyCompatibilityDisposition.CURRENT_ONLY_FAIL_CLOSED
    )
    no_prior = sum(
        1
        for o in obligations
        if o.disposition == LegacyCompatibilityDisposition.NO_PRIOR_DURABLE_REPRESENTATION
    )
    deferred = sum(
        1
        for o in obligations
        if o.disposition == LegacyCompatibilityDisposition.DEFERRED_TO_EVENT_EVOLUTION
    )

    dup_migration = len(_duplicate_migration_owners())

    return PersistedR2R2Metrics(
        legacy_compatibility_obligations=len(obligations),
        explicit_migrations=explicit_migrations,
        explicit_legacy_readers=explicit_readers,
        current_only_fail_closed=current_only,
        no_prior_version=no_prior,
        deferred_event_evolution=deferred,
        unclassified_legacy_obligations=unclassified,
        migration_owner_missing=migration_owner_missing,
        duplicate_migration_authority=dup_migration,
        implicit_compatibility_branches=implicit,
    )


def persisted_r2_r2_gate_passes() -> bool:
    metrics = compute_persisted_r2_r2_metrics()
    if metrics.unclassified_legacy_obligations != 0:
        return False
    if metrics.migration_owner_missing != 0:
        return False
    if metrics.duplicate_migration_authority != 0:
        return False
    if metrics.implicit_compatibility_branches != 0:
        return False
    for family_id in legacy_r2_r2_inventory_family_ids():
        family = _family_by_id(family_id)
        if family.migration_disposition in (
            MigrationDisposition.MIGRATION_REQUIRED_BUT_MISSING,
            MigrationDisposition.UNKNOWN,
        ):
            return False
    return True


def legacy_r2_r2_inventory_family_ids() -> tuple[str, ...]:
    from tests.qualification.compat_x._compat_x_legacy_obligations import (
        legacy_r2_r2_remediation_family_ids,
    )

    return legacy_r2_r2_remediation_family_ids()


def inventory_matches_legacy_obligations() -> bool:
    disposition_map = {
        LegacyCompatibilityDisposition.EXPLICIT_MIGRATION: MigrationDisposition.EXPLICIT_MIGRATION,
        LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER: MigrationDisposition.EXPLICIT_LEGACY_READER,
        LegacyCompatibilityDisposition.CURRENT_ONLY_FAIL_CLOSED: MigrationDisposition.CURRENT_ONLY_FAIL_CLOSED,
        LegacyCompatibilityDisposition.NO_PRIOR_DURABLE_REPRESENTATION: MigrationDisposition.CURRENT_ONLY_FAIL_CLOSED,
        LegacyCompatibilityDisposition.DEFERRED_TO_EVENT_EVOLUTION: MigrationDisposition.CURRENT_ONLY_FAIL_CLOSED,
    }
    primary: dict[str, LegacyCompatibilityDisposition] = {}
    for obligation in legacy_persisted_compatibility_obligations():
        if obligation.disposition == LegacyCompatibilityDisposition.DEFERRED_TO_EVENT_EVOLUTION:
            continue
        primary.setdefault(obligation.family_id, obligation.disposition)
    for family_id, legacy_disp in primary.items():
        family = _family_by_id(family_id)
        expected = disposition_map[legacy_disp]
        if family.migration_disposition != expected:
            return False
    return True
