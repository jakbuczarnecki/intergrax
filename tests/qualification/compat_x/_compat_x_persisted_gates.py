# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-P0 qualification gates for persisted schema / migration ownership."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.compat_x._compat_x_persisted_discovery import (
    build_persisted_mechanical_discovery_report,
    inventory_family_ids,
    mechanical_closed_world_family_ids,
    persisted_closed_world_parity_holds,
)
from tests.qualification.compat_x._compat_x_persisted_historical_five import (
    historical_five_contradiction_count,
    historical_persisted_five_reconciliation,
)
from tests.qualification.compat_x._compat_x_persisted_inventory import PERSISTED_SCHEMA_FAMILIES
from tests.qualification.compat_x._compat_x_persisted_types import (
    MigrationDisposition,
    PersistedR2P0Metrics,
    PersistedVersionDisposition,
)
from tests.qualification.compat_x._compat_x_versioning_gates import build_r1_qualification_report

_QUALIFICATION_MODULE_PREFIX = "tests/qualification/compat_x/"


@dataclass(frozen=True, slots=True)
class PersistedR2P0QualificationReport:
    metrics: PersistedR2P0Metrics
    closed_world_parity: bool
    historical_five_reconciled: bool
    historical_contradictions: int
    r1_public_stable_noncompliant_count: int
    writer_without_reader_violations: tuple[str, ...]
    reader_without_writer_violations: tuple[str, ...]
    duplicate_version_authority: tuple[str, ...]
    duplicate_migration_authority: tuple[str, ...]
    qualification_as_version_authority: int
    service_misclassified_as_payload: int


def compute_persisted_r2_p0_metrics() -> PersistedR2P0Metrics:
    families = PERSISTED_SCHEMA_FAMILIES
    writers = {p for f in families for p in f.writer_paths}
    readers = {p for f in families for p in f.reader_paths}
    versioned = sum(
        1
        for f in families
        if f.version_disposition == PersistedVersionDisposition.EXPLICIT_PAYLOAD_VERSION
        and f.current_version is not None
    )
    inherited = sum(
        1
        for f in families
        if f.version_disposition == PersistedVersionDisposition.INHERITED_ENVELOPE_VERSION
    )
    missing = sum(
        1 for f in families if f.version_disposition == PersistedVersionDisposition.MISSING_VERSION
    )
    migration_required = sum(
        1
        for f in families
        if f.migration_disposition
        in (
            MigrationDisposition.MIGRATION_REQUIRED_BUT_MISSING,
            MigrationDisposition.UNKNOWN,
        )
    )
    migration_owner_missing = sum(
        1
        for f in families
        if f.migration_disposition
        in (
            MigrationDisposition.MIGRATION_REQUIRED_BUT_MISSING,
            MigrationDisposition.UNKNOWN,
        )
        and f.migration_owner_path is None
    )
    discovery = build_persisted_mechanical_discovery_report()
    dup_version = _duplicate_version_authority_keys()
    dup_migration = _duplicate_migration_authority_keys()
    return PersistedR2P0Metrics(
        family_count=len(families),
        writer_path_count=len(writers),
        reader_path_count=len(readers),
        versioned_family_count=versioned,
        inherited_version_family_count=inherited,
        missing_version_family_count=missing,
        migration_required_count=migration_required,
        migration_owner_missing_count=migration_owner_missing,
        unknown_unclassified_count=len(discovery.unclassified_candidate_ids),
        duplicate_version_authority_count=len(dup_version),
        duplicate_migration_authority_count=len(dup_migration),
    )


def _duplicate_version_authority_keys() -> tuple[str, ...]:
    """Same semantic family_id must not list two version owners."""
    owners_by_family: dict[str, str] = {}
    duplicates: list[str] = []
    for family in PERSISTED_SCHEMA_FAMILIES:
        owner = family.version_owner_path
        if owner is None:
            continue
        if family.version_disposition in (
            PersistedVersionDisposition.NOT_A_PERSISTED_SCHEMA,
            PersistedVersionDisposition.STORE_CONTAINER_VERSION,
        ):
            continue
        prior = owners_by_family.get(family.family_id)
        if prior is not None and prior != owner:
            duplicates.append(f"{family.family_id}:{prior}|{owner}")
        else:
            owners_by_family[family.family_id] = owner
    return tuple(sorted(duplicates))


def _duplicate_migration_authority_keys() -> tuple[str, ...]:
    """Same family_id must not list two migration owners (store codec reuse is allowed across families)."""
    owners_by_family: dict[str, str] = {}
    duplicates: list[str] = []
    for family in PERSISTED_SCHEMA_FAMILIES:
        owner = family.migration_owner_path
        if owner is None:
            continue
        prior = owners_by_family.get(family.family_id)
        if prior is not None and prior != owner:
            duplicates.append(f"{family.family_id}:{prior}|{owner}")
        else:
            owners_by_family[family.family_id] = owner
    return tuple(sorted(duplicates))


def _writer_reader_violations() -> tuple[tuple[str, ...], tuple[str, ...]]:
    writer_only: list[str] = []
    reader_only: list[str] = []
    for family in PERSISTED_SCHEMA_FAMILIES:
        if family.version_disposition == PersistedVersionDisposition.NOT_A_PERSISTED_SCHEMA:
            continue
        if not family.reader_paths and family.writer_paths:
            writer_only.append(family.family_id)
        if not family.writer_paths and family.reader_paths:
            reader_only.append(family.family_id)
    return tuple(sorted(writer_only)), tuple(sorted(reader_only))


def build_persisted_r2_p0_qualification_report() -> PersistedR2P0QualificationReport:
    r1 = build_r1_qualification_report()
    writer_only, reader_only = _writer_reader_violations()
    qual_as_authority = sum(
        1
        for f in PERSISTED_SCHEMA_FAMILIES
        if f.version_owner_path is not None
        and f.version_owner_path.startswith(_QUALIFICATION_MODULE_PREFIX)
    )
    return PersistedR2P0QualificationReport(
        metrics=compute_persisted_r2_p0_metrics(),
        closed_world_parity=persisted_closed_world_parity_holds(),
        historical_five_reconciled=len(historical_persisted_five_reconciliation()) == 5,
        historical_contradictions=historical_five_contradiction_count(),
        r1_public_stable_noncompliant_count=r1.public_stable_noncompliant_version_identity_count,
        writer_without_reader_violations=writer_only,
        reader_without_writer_violations=reader_only,
        duplicate_version_authority=_duplicate_version_authority_keys(),
        duplicate_migration_authority=_duplicate_migration_authority_keys(),
        qualification_as_version_authority=qual_as_authority,
        service_misclassified_as_payload=0,
    )


def persisted_r2_p0_inventory_gate_passes() -> bool:
    report = build_persisted_r2_p0_qualification_report()
    if not report.closed_world_parity:
        return False
    if mechanical_closed_world_family_ids() != inventory_family_ids():
        return False
    if report.metrics.unknown_unclassified_count != 0:
        return False
    if report.metrics.duplicate_version_authority_count != 0:
        return False
    if report.metrics.duplicate_migration_authority_count != 0:
        return False
    if report.historical_contradictions != 0:
        return False
    if not report.historical_five_reconciled:
        return False
    if report.writer_without_reader_violations or report.reader_without_writer_violations:
        return False
    if report.qualification_as_version_authority != 0:
        return False
    if report.r1_public_stable_noncompliant_count != 0:
        return False
    return True
