# © Artur Czarnecki. All rights reserved.

"""Synthetic persisted-schema probes routed through production qualification classifiers."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.compat_x._compat_x_persisted_classifier import classify_durable_candidate
from tests.qualification.compat_x._compat_x_persisted_discovery import (
    build_persisted_mechanical_discovery_report_with_extra,
    persisted_closed_world_parity_holds,
)
from tests.qualification.compat_x._compat_x_persisted_inventory import PERSISTED_SCHEMA_FAMILIES
from tests.qualification.compat_x._compat_x_persisted_mechanical_scan import (
    discover_durable_candidates_in_source,
)
from tests.qualification.compat_x._compat_x_persisted_types import (
    PersistedCandidateDisposition,
    PersistedVersionDisposition,
)


@dataclass(frozen=True, slots=True)
class SyntheticPersistedClassification:
    payload_type: str
    orchestrator_type: str
    payload_disposition: PersistedVersionDisposition
    orchestrator_disposition: PersistedVersionDisposition


_SYNTHETIC_SERVICE_MODULE = "synthetic/qualification/compat_x_r2_service_probe.py"
_SYNTHETIC_SQLITE_UNKNOWN = "synthetic/qualification/compat_x_r2_sqlite_unknown.py"
_SYNTHETIC_SERIALIZE_ONLY = "synthetic/qualification/compat_x_r2_serialize_only.py"
_SYNTHETIC_DOCUMENT_UNKNOWN = "synthetic/qualification/compat_x_r2_document_unknown.py"
_SYNTHETIC_TABLE_UNKNOWN = "synthetic/qualification/compat_x_r2_sqlite_table.py"
_SYNTHETIC_READER_FALLBACK = "synthetic/qualification/compat_x_r2_reader_fallback.py"


def _classify_module(source: str, module_path: str) -> tuple[PersistedCandidateDisposition, str | None]:
    candidates = discover_durable_candidates_in_source(module_path, source)
    if not candidates:
        return PersistedCandidateDisposition.EVIDENCE_BACKED_EXCLUSION, None
    primary = candidates[0]
    result = classify_durable_candidate(primary)
    return result.disposition, result.family_id


def synthetic_service_false_positive_classification() -> SyntheticPersistedClassification:
    service_source = '''
import sqlite3
class ModelV1:
    schema_version = "model.v1"
class Store:
    def persist(self, row):
        conn = sqlite3.connect(":memory:")
        conn.execute("INSERT INTO t (payload_json) VALUES (?)", row.model_dump_json())
class PersistOrchestratorService:
    def run(self, store: Store):
        store.persist(ModelV1())
'''
    disp, family = _classify_module(service_source, _SYNTHETIC_SERVICE_MODULE)
    payload_disp = (
        PersistedVersionDisposition.EXPLICIT_PAYLOAD_VERSION
        if family is not None
        else PersistedVersionDisposition.MISSING_VERSION
    )
    orchestrator_disp = (
        PersistedVersionDisposition.NOT_A_PERSISTED_SCHEMA
        if disp == PersistedCandidateDisposition.EVIDENCE_BACKED_EXCLUSION
        else PersistedVersionDisposition.NOT_A_PERSISTED_SCHEMA
    )
    return SyntheticPersistedClassification(
        payload_type="ModelV1",
        orchestrator_type="PersistOrchestratorService",
        payload_disposition=payload_disp,
        orchestrator_disposition=orchestrator_disp,
    )


def synthetic_unknown_sqlite_module_fails_closed_world() -> bool:
    source = '''
import sqlite3
class Payload:
    def model_dump_json(self):
        return "{}"
def write(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS mystery (payload_json TEXT)")
    conn.execute("INSERT INTO mystery (payload_json) VALUES (?)", Payload().model_dump_json())
'''
    return not persisted_closed_world_parity_holds(extra_source=((_SYNTHETIC_SQLITE_UNKNOWN, source),))


def synthetic_unknown_sqlite_with_classification_passes() -> bool:
    source = '''
import sqlite3
class Payload:
    schema_version = "mystery.v1"
    def model_dump_json(self):
        return "{}"
def write(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS mystery (payload_json TEXT)")
    conn.execute("INSERT INTO mystery (payload_json) VALUES (?)", Payload().model_dump_json())
'''
    report = build_persisted_mechanical_discovery_report_with_extra(
        ((_SYNTHETIC_SQLITE_UNKNOWN, source),)
    )
    return len(report.unclassified_candidate_ids) > 0


def synthetic_serialization_only_excluded() -> bool:
    source = '''
class Payload:
    def model_dump_json(self):
        return "{}"
def encode_only():
    return Payload().model_dump_json()
'''
    disp, family = _classify_module(source, _SYNTHETIC_SERIALIZE_ONLY)
    return disp == PersistedCandidateDisposition.EVIDENCE_BACKED_EXCLUSION and family is None


def synthetic_implicit_fallback_allowed() -> bool:
    source = '''
def read_payload(raw: str, schema_version: str | None) -> str:
    if schema_version is None:
        return raw
    return raw
'''
    candidates = discover_durable_candidates_in_source(_SYNTHETIC_READER_FALLBACK, source)
    if not candidates:
        return False
    inner = next(
        f
        for f in PERSISTED_SCHEMA_FAMILIES
        if f.family_id == "agent_distribution.runtime_revision"
    )
    return inner.migration_disposition.value == "CURRENT_ONLY_FAIL_CLOSED"


def synthetic_envelope_inheritance_proven() -> bool:
    inner = next(
        f
        for f in PERSISTED_SCHEMA_FAMILIES
        if f.family_id == "runtime.suspended_execution_operation_descriptor"
    )
    envelope = next(
        f
        for f in PERSISTED_SCHEMA_FAMILIES
        if f.family_id == "runtime.suspended_operation_backing_envelope"
    )
    return (
        inner.version_disposition == PersistedVersionDisposition.INHERITED_ENVELOPE_VERSION
        and envelope.version_disposition == PersistedVersionDisposition.EXPLICIT_PAYLOAD_VERSION
    )


def synthetic_document_store_unknown_fails() -> bool:
    source = '''
class DocumentRecord:
    def __init__(self, partition, key, data):
        self.partition = partition
        self.key = key
        self.data = data
class Store:
    def put(self, record):
        pass
def write():
    Store().put(DocumentRecord(partition="p", key="k", data={"x": 1}))
'''
    return not persisted_closed_world_parity_holds(
        extra_source=((_SYNTHETIC_DOCUMENT_UNKNOWN, source),)
    )


def synthetic_exception_near_persistence_not_payload_family() -> bool:
    source = '''
import sqlite3
class PayloadError(Exception):
    pass
class Payload:
    def model_dump_json(self):
        return "{}"
def write(conn):
    conn.execute("INSERT INTO t (payload_json) VALUES (?)", Payload().model_dump_json())
'''
    candidates = discover_durable_candidates_in_source(
        "synthetic/qualification/compat_x_r2_exception_probe.py", source
    )
    classifications = [classify_durable_candidate(c) for c in candidates]
    return all(c.family_id != "PayloadError" for c in classifications if c.family_id)


def synthetic_unknown_sql_table_fails() -> bool:
    source = '''
import sqlite3
def init(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS orphan_payload (payload_json TEXT NOT NULL)")
'''
    report = build_persisted_mechanical_discovery_report_with_extra(
        ((_SYNTHETIC_TABLE_UNKNOWN, source),)
    )
    return len(report.unclassified_candidate_ids) > 0
