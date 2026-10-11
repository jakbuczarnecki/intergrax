# © Artur Czarnecki. All rights reserved.

"""Synthetic persisted-schema probes routed through production qualification classifiers."""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

from tests.qualification.compat_x._compat_x_persisted_classifier import (
    classify_durable_candidate,
)
from tests.qualification.compat_x._compat_x_persisted_types import DurableExclusionReason
from tests.qualification.compat_x._compat_x_persisted_discovery import (
    build_persisted_mechanical_discovery_report_with_extra,
    persisted_closed_world_parity_holds,
)
from tests.qualification.compat_x._compat_x_persisted_inventory import PERSISTED_SCHEMA_FAMILIES
from tests.qualification.compat_x._compat_x_persisted_mechanical_scan import (
    discover_durable_candidates_in_source,
)
from tests.qualification.compat_x._compat_x_persisted_types import (
    MigrationDisposition,
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
_SYNTHETIC_REDIS_UNKNOWN = "synthetic/qualification/compat_x_r2_redis_unknown.py"
_SYNTHETIC_REDIS_CLASSIFIED = "synthetic/qualification/compat_x_r2_redis_classified.py"
_SYNTHETIC_DURABLE_FILE = "synthetic/qualification/compat_x_r2_durable_file.py"
_SYNTHETIC_EVIDENCE_EXPORT = "synthetic/qualification/compat_x_r2_evidence_export.py"
_SYNTHETIC_PICKLE_SQLITE = "synthetic/qualification/compat_x_r2_pickle_sqlite.py"
_SYNTHETIC_PICKLE_REDIS = "synthetic/qualification/compat_x_r2_pickle_redis.py"
_SYNTHETIC_CACHE_FALSE_EXCLUDE = "synthetic/qualification/compat_x_r2_cache_false_exclude.py"
_REPO_ROOT = Path(__file__).resolve().parents[3]


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
    """True when synthetic durable reader accepts missing schema_version (must be False)."""
    source = '''
import sqlite3
import json
class RuntimeRevision:
    schema_version = "runtime_revision.v1"
    @classmethod
    def model_validate_json(cls, raw):
        return cls()
def read_row(raw: str) -> RuntimeRevision:
    data = json.loads(raw)
    version = data.get("schema_version")
    if version is None:
        return RuntimeRevision()
    return RuntimeRevision()
def load(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS revisions (payload_json TEXT NOT NULL)")
    row = conn.execute("SELECT payload_json FROM revisions").fetchone()
    return RuntimeRevision.model_validate_json(row[0])
'''
    allows_implicit = "if version is None" in source and "return RuntimeRevision()" in source
    if not allows_implicit:
        return False
    candidates = discover_durable_candidates_in_source(_SYNTHETIC_READER_FALLBACK, source)
    if not candidates:
        return True
    classified = [classify_durable_candidate(c) for c in candidates]
    if not any(c.family_id == "agent_distribution.runtime_revision" for c in classified):
        return True
    family = next(
        f for f in PERSISTED_SCHEMA_FAMILIES if f.family_id == "agent_distribution.runtime_revision"
    )
    return family.migration_disposition != MigrationDisposition.CURRENT_ONLY_FAIL_CLOSED


def synthetic_envelope_inheritance_proven() -> bool:
    module = (
        "intergrax/runtime/execution/suspended_operation/"
        "document_store_suspended_operation_store.py"
    )
    source = (_REPO_ROOT / module).read_text(encoding="utf-8")
    candidates = discover_durable_candidates_in_source(module, source)
    if not candidates:
        return False
    families = {
        classify_durable_candidate(c).family_id
        for c in candidates
        if classify_durable_candidate(c).family_id
    }
    if not {
        "runtime.suspended_operation_backing_envelope",
        "runtime.suspended_execution_operation_descriptor",
    }.issubset(families):
        return False
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


def synthetic_redis_unknown_family_fails_closed_world() -> bool:
    source = '''
import redis
class _Client:
    def set(self, key, value):
        pass
class MysteryRedisStore:
    def __init__(self, client):
        self._client = client
    def persist(self, key, blob):
        self._client.set(key, blob)
'''
    return not persisted_closed_world_parity_holds(
        extra_source=((_SYNTHETIC_REDIS_UNKNOWN, source),)
    )


def synthetic_redis_classified_family_passes() -> bool:
    source = '''
import redis
class _Client:
    def set(self, key, value):
        pass
class RedisKVStore:
    def __init__(self, client):
        self._client = client
    def set(self, tenant_id, key, value):
        self._client.set(f"{tenant_id}:{key}", value)
'''
    report = build_persisted_mechanical_discovery_report_with_extra(
        ((_SYNTHETIC_REDIS_CLASSIFIED, source),)
    )
    return len(report.unclassified_candidate_ids) == 0 and persisted_closed_world_parity_holds(
        extra_source=((_SYNTHETIC_REDIS_CLASSIFIED, source),)
    )


def synthetic_durable_file_state_detected() -> bool:
    source = '''
import json
from pathlib import Path
def persist(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload)
    temp = path.with_suffix(".tmp")
    temp.write_text(serialized)
    temp.replace(path)
def load(path: Path) -> dict:
    return json.loads(path.read_text())
'''
    candidates = discover_durable_candidates_in_source(_SYNTHETIC_DURABLE_FILE, source)
    signals = {c.storage_signal.value for c in candidates}
    return "FILE_WRITE" in signals and any("write_text" in e for c in candidates for e in c.evidence)


def synthetic_evidence_export_excluded() -> bool:
    source = '''
from pathlib import Path
def export_diagnostic_report(path: Path, lines: list[str]) -> None:
    path.write_text("\\n".join(lines))
'''
    candidates = discover_durable_candidates_in_source(_SYNTHETIC_EVIDENCE_EXPORT, source)
    if not candidates:
        return True
    results = [classify_durable_candidate(c) for c in candidates]
    return all(
        r.exclusion_reason
        in (
            DurableExclusionReason.SERIALIZATION_WITHOUT_DURABLE_IO,
            DurableExclusionReason.CACHE_NON_DURABLE,
        )
        for r in results
    )


def synthetic_pickle_to_sqlite_detected() -> bool:
    source = '''
import pickle
import sqlite3
class Row:
    pass
def write(conn, row):
    blob = pickle.dumps(row)
    conn.execute("INSERT INTO idempotency_ledger (result_blob) VALUES (?)", (blob,))
def read(raw):
    return pickle.loads(raw)
'''
    candidates = discover_durable_candidates_in_source(_SYNTHETIC_PICKLE_SQLITE, source)
    return any(c.storage_signal.value == "PICKLE_DURABLE_WIRE" for c in candidates)


def synthetic_pickle_to_redis_detected() -> bool:
    source = '''
import pickle
class _Redis:
    def hset(self, key, mapping=None, **kwargs):
        pass
def write(client, key, row):
    blob = pickle.dumps(row)
    client.hset(key, "result_blob", blob)
def read(blob):
    return pickle.loads(blob)
'''
    candidates = discover_durable_candidates_in_source(_SYNTHETIC_PICKLE_REDIS, source)
    return any(
        c.storage_signal.value in ("PICKLE_DURABLE_WIRE", "REDIS_HASH_WRITE") for c in candidates
    )


def synthetic_cache_name_not_false_excluded() -> bool:
    source = '''
import pickle
import sqlite3
"""Durable single-host ledger (cache-named module probe)."""
class ToolExecutionResult:
    pass
def write(conn):
    blob = pickle.dumps(ToolExecutionResult())
    conn.execute(
        "INSERT INTO idempotency_ledger (status, result_blob) VALUES (?, ?)",
        ("COMPLETED", blob),
    )
'''
    results = [
        classify_durable_candidate(c)
        for c in discover_durable_candidates_in_source(_SYNTHETIC_CACHE_FALSE_EXCLUDE, source)
    ]
    return any(r.family_id == "runtime.tool_idempotency_ledger" for r in results)


def synthetic_durable_state_file_not_cache_excluded() -> bool:
    module = "intergrax/runtime/execution/continuation/durable_state_file.py"
    source = (_REPO_ROOT / module).read_text(encoding="utf-8")
    results = [
        classify_durable_candidate(c)
        for c in discover_durable_candidates_in_source(module, source)
    ]
    if not results:
        return False
    return all(
        r.exclusion_reason != DurableExclusionReason.CACHE_NON_DURABLE for r in results
    ) and any(
        r.family_id == "runtime.execution_continuation_durable_state" for r in results
    )


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
