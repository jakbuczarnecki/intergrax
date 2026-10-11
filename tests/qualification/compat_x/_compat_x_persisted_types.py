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
    REDIS_BYTES = "REDIS_BYTES"
    REDIS_HASH_LEDGER = "REDIS_HASH_LEDGER"
    PICKLE_BLOB_LEDGER = "PICKLE_BLOB_LEDGER"


class PersistedVersionDisposition(StrEnum):
    EXPLICIT_PAYLOAD_VERSION = "EXPLICIT_PAYLOAD_VERSION"
    INHERITED_ENVELOPE_VERSION = "INHERITED_ENVELOPE_VERSION"
    STORE_CONTAINER_VERSION = "STORE_CONTAINER_VERSION"
    MISSING_VERSION = "MISSING_VERSION"
    NOT_A_PERSISTED_SCHEMA = "NOT_A_PERSISTED_SCHEMA"
    OPAQUE_CARRIER_NOT_SEMANTIC_PAYLOAD = "OPAQUE_CARRIER_NOT_SEMANTIC_PAYLOAD"


class MigrationDisposition(StrEnum):
    NO_PRIOR_VERSION = "NO_PRIOR_VERSION"
    EXPLICIT_MIGRATION = "EXPLICIT_MIGRATION"
    EXPLICIT_LEGACY_READER = "EXPLICIT_LEGACY_READER"
    CURRENT_ONLY_FAIL_CLOSED = "CURRENT_ONLY_FAIL_CLOSED"
    MIGRATION_REQUIRED_BUT_MISSING = "MIGRATION_REQUIRED_BUT_MISSING"
    UNKNOWN = "UNKNOWN"


class LegacyCompatibilityDisposition(StrEnum):
    EXPLICIT_MIGRATION = "EXPLICIT_MIGRATION"
    EXPLICIT_LEGACY_READER = "EXPLICIT_LEGACY_READER"
    CURRENT_ONLY_FAIL_CLOSED = "CURRENT_ONLY_FAIL_CLOSED"
    NO_PRIOR_DURABLE_REPRESENTATION = "NO_PRIOR_DURABLE_REPRESENTATION"
    DEFERRED_TO_EVENT_EVOLUTION = "DEFERRED_TO_EVENT_EVOLUTION"


class DurableStorageSignal(StrEnum):
    SQLITE_DML = "SQLITE_DML"
    SQLITE_PAYLOAD_TABLE = "SQLITE_PAYLOAD_TABLE"
    DOCUMENT_STORE_WRITE = "DOCUMENT_STORE_WRITE"
    FILE_WRITE = "FILE_WRITE"
    POSTGRESQL_DML = "POSTGRESQL_DML"
    EXPLICIT_PERSIST_API = "EXPLICIT_PERSIST_API"
    WIRE_TO_DURABLE_STORE = "WIRE_TO_DURABLE_STORE"
    REDIS_KV_WRITE = "REDIS_KV_WRITE"
    REDIS_HASH_WRITE = "REDIS_HASH_WRITE"
    REDIS_SCRIPTED_STATE = "REDIS_SCRIPTED_STATE"
    REDIS_READ = "REDIS_READ"
    PICKLE_DURABLE_WIRE = "PICKLE_DURABLE_WIRE"


class DurableAccessDirection(StrEnum):
    WRITE = "WRITE"
    READ = "READ"
    READ_WRITE = "READ_WRITE"


class PersistedCandidateDisposition(StrEnum):
    PERSISTED_FAMILY_MEMBER = "PERSISTED_FAMILY_MEMBER"
    EVIDENCE_BACKED_EXCLUSION = "EVIDENCE_BACKED_EXCLUSION"


class DurableExclusionReason(StrEnum):
    EPHEMERAL_IN_MEMORY_ONLY = "EPHEMERAL_IN_MEMORY_ONLY"
    ORCHESTRATOR_SERVICE_NOT_PAYLOAD = "ORCHESTRATOR_SERVICE_NOT_PAYLOAD"
    EXCEPTION_NOT_PAYLOAD = "EXCEPTION_NOT_PAYLOAD"
    PROTOCOL_OR_INTERFACE_ONLY = "PROTOCOL_OR_INTERFACE_ONLY"
    SERIALIZATION_WITHOUT_DURABLE_IO = "SERIALIZATION_WITHOUT_DURABLE_IO"
    SQL_METADATA_NOT_SEMANTIC_PAYLOAD = "SQL_METADATA_NOT_SEMANTIC_PAYLOAD"
    TRANSPORT_ONLY_ENCODING = "TRANSPORT_ONLY_ENCODING"
    CACHE_NON_DURABLE = "CACHE_NON_DURABLE"
    TEST_OR_LAB_ONLY = "TEST_OR_LAB_ONLY"


class HistoricalFindingAssessment(StrEnum):
    FALSE_POSITIVE_SERVICE_OR_ORCHESTRATOR = "FALSE_POSITIVE_SERVICE_OR_ORCHESTRATOR"
    FALSE_POSITIVE_EXCEPTION = "FALSE_POSITIVE_EXCEPTION"
    FALSE_POSITIVE_PUBLISHER_PROTOCOL = "FALSE_POSITIVE_PUBLISHER_PROTOCOL"
    REAL_DURABLE_BOUNDARY_STORE = "REAL_DURABLE_BOUNDARY_STORE"
    PRE_AUDIT_CONTRADICTION = "PRE_AUDIT_CONTRADICTION"


@dataclass(frozen=True, slots=True)
class LegacyPersistedCompatibilityObligation:
    family_id: str
    legacy_shape_id: str
    current_version: str
    disposition: LegacyCompatibilityDisposition
    semantic_owner_path: str
    migration_owner_path: str | None
    reader_path: str
    evidence_paths: tuple[str, ...]


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
class DurableBoundaryCandidate:
    candidate_id: str
    module_path: str
    class_name: str | None
    function_name: str | None
    storage_signal: DurableStorageSignal
    write_or_read: DurableAccessDirection
    payload_symbol: str | None
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DurableCandidateClassification:
    candidate: DurableBoundaryCandidate
    disposition: PersistedCandidateDisposition
    family_id: str | None
    exclusion_reason: DurableExclusionReason | None
    classification_evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PersistedMechanicalDiscoveryReport:
    discovery_roots: tuple[str, ...]
    raw_candidates: tuple[DurableBoundaryCandidate, ...]
    classifications: tuple[DurableCandidateClassification, ...]
    unclassified_candidate_ids: frozenset[str]
    semantic_persisted_family_ids: frozenset[str]
    exclusion_count: int
    persisted_member_count: int


@dataclass(frozen=True, slots=True)
class PersistedRemediationWave:
    wave_id: str
    title: str
    items: tuple[PersistedRemediationItem, ...]
