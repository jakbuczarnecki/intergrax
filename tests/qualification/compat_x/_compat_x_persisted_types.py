# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-P0 persisted schema family qualification types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PersistedStorageKind(StrEnum):
    SQLITE_JSON_PAYLOAD = "SQLITE_JSON_PAYLOAD"
    SQLITE_ROW_COLUMNS = "SQLITE_ROW_COLUMNS"
    DOCUMENT_STORE_PARTITION = "DOCUMENT_STORE_PARTITION"
    POSTGRESQL_ROW = "POSTGRESQL_ROW"
    IN_MEMORY_DURABLE_STATE = "IN_MEMORY_DURABLE_STATE"
    FILE_JSON = "FILE_JSON"


class PersistedVersionDisposition(StrEnum):
    EXPLICIT_PAYLOAD_VERSION = "EXPLICIT_PAYLOAD_VERSION"
    INHERITED_ENVELOPE_VERSION = "INHERITED_ENVELOPE_VERSION"
    STORE_CONTAINER_VERSION = "STORE_CONTAINER_VERSION"
    MISSING_VERSION = "MISSING_VERSION"
    NOT_A_PERSISTED_SCHEMA = "NOT_A_PERSISTED_SCHEMA"


class MigrationDisposition(StrEnum):
    NO_PRIOR_VERSION = "NO_PRIOR_VERSION"
    EXPLICIT_MIGRATION = "EXPLICIT_MIGRATION"
    EXPLICIT_LEGACY_READER = "EXPLICIT_LEGACY_READER"
    CURRENT_ONLY_FAIL_CLOSED = "CURRENT_ONLY_FAIL_CLOSED"
    MIGRATION_REQUIRED_BUT_MISSING = "MIGRATION_REQUIRED_BUT_MISSING"
    UNKNOWN = "UNKNOWN"


class HistoricalFindingAssessment(StrEnum):
    FALSE_POSITIVE_SERVICE_OR_ORCHESTRATOR = "FALSE_POSITIVE_SERVICE_OR_ORCHESTRATOR"
    FALSE_POSITIVE_EXCEPTION = "FALSE_POSITIVE_EXCEPTION"
    FALSE_POSITIVE_PUBLISHER_PROTOCOL = "FALSE_POSITIVE_PUBLISHER_PROTOCOL"
    REAL_DURABLE_BOUNDARY_STORE = "REAL_DURABLE_BOUNDARY_STORE"
    PRE_AUDIT_CONTRADICTION = "PRE_AUDIT_CONTRADICTION"


@dataclass(frozen=True, slots=True)
class PersistedSchemaFamily:
    family_id: str
    canonical_payload_type: str
    owner_module_path: str
    writer_paths: tuple[str, ...]
    reader_paths: tuple[str, ...]
    storage_kind: PersistedStorageKind
    version_disposition: PersistedVersionDisposition
    current_version: str | None
    version_owner_path: str | None
    migration_disposition: MigrationDisposition
    migration_owner_path: str | None
    evidence_paths: tuple[str, ...]
    frz_cmp_tags: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class HistoricalPersistedFindingReconciliation:
    finding_class: str
    module_path: str
    assessment: HistoricalFindingAssessment
    actual_persisted_payload_type: str | None
    actual_current_version: str | None
    evidence_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PersistedR2P0Metrics:
    family_count: int
    writer_path_count: int
    reader_path_count: int
    versioned_family_count: int
    inherited_version_family_count: int
    missing_version_family_count: int
    migration_required_count: int
    migration_owner_missing_count: int
    unknown_unclassified_count: int
    duplicate_version_authority_count: int
    duplicate_migration_authority_count: int


@dataclass(frozen=True, slots=True)
class PersistedRemediationItem:
    family_id: str
    canonical_owner: str
    current_writer: str
    current_reader: str
    format_version_state: str
    required_change: str
    migration_behavior_required: str
    tests_required: str
    frz_mapping: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PersistedRemediationWave:
    wave_id: str
    title: str
    items: tuple[PersistedRemediationItem, ...]
