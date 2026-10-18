# © Artur Czarnecki. All rights reserved.

"""Dedicated SQLite GCA host store bundle (PROD-Q-R1-R7-Q GCA durable recovery)."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from governed_contractor_application.host.governed_external_work_host_stores import (
    GovernedExternalWorkHostStoreBundle,
)
from governed_contractor_application.host.lifecycle_states import (
    GovernedExternalWorkHostState,
)
from governed_contractor_application.host.stores import (
    GovernedExecutionCreateFollowOnConflictError,
)
from intergrax.contracts.execution_evidence.receipt import (
    ExecutionEvidenceProofReceipt,
    parse_execution_evidence_proof_receipt_json,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.governed_execution_result import (
    AnyGovernedExecutionResult,
    parse_governed_execution_result_json,
)
from intergrax.contracts.governed_external_work_create_follow_on import (
    GovernedExternalWorkCreateFollowOnRecordV1,
    follow_on_records_equivalent,
    parse_governed_external_work_create_follow_on_record_json,
)
from intergrax.contracts.governed_external_work_host_store_lifecycle import (
    OwnedGovernedExternalWorkHostStoreBundle,
)
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle
from intergrax.runtime.attestation.canonical_json import canonical_json_text

_SCHEMA_META_KEY = "schema_version"
_UNCERTAIN_WRITE_META_KEY = "durable_write_uncertain"
_SCHEMA_VERSION = "1"

_REQUIRED_TABLES = frozenset(
    {
        "gca_host_store_meta",
        "gca_execution_results",
        "gca_execution_states",
        "gca_execution_events",
        "gca_create_follow_on",
        "gca_proof_receipts",
        "gca_policy_bundles",
        "gca_continuations",
    },
)


class GovernedExternalWorkHostPersistenceError(ValueError):
    """Fail-closed GCA host SQLite persistence error."""


class _SqliteGovernedExternalWorkHostDatabase:
    __slots__ = (
        "_closed",
        "_db_path",
        "_had_existing_content",
        "_lock",
        "_writes_uncertain",
    )

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._closed = False
        self._writes_uncertain = False
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._had_existing_content = (
            self._db_path.exists() and self._db_path.stat().st_size > 0
        )
        self._bootstrap_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                conn = sqlite3.connect(self._db_path)
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                finally:
                    conn.close()
            except sqlite3.Error:
                pass
            self._closed = True

    def _ensure_operational(self) -> None:
        if self._closed:
            raise GovernedExternalWorkHostPersistenceError("gca host store is closed")
        if self._writes_uncertain:
            raise GovernedExternalWorkHostPersistenceError(
                "gca host store transaction outcome uncertain",
            )

    def _connect(self) -> sqlite3.Connection:
        self._ensure_operational()
        try:
            conn = sqlite3.connect(self._db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=FULL;")
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            raise GovernedExternalWorkHostPersistenceError(str(exc)) from exc
        return conn

    @contextmanager
    def _managed_connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connect()
            try:
                yield conn
            finally:
                conn.close()

    @staticmethod
    def _list_user_tables(conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
        ).fetchall()
        return {row[0] for row in rows}

    @staticmethod
    def _read_schema_version(conn: sqlite3.Connection) -> str | None:
        try:
            row = conn.execute(
                "SELECT value FROM gca_host_store_meta WHERE key = ?",
                (_SCHEMA_META_KEY,),
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        return str(row["value"])

    @staticmethod
    def _read_uncertain_write_flag(conn: sqlite3.Connection) -> bool:
        try:
            row = conn.execute(
                "SELECT value FROM gca_host_store_meta WHERE key = ?",
                (_UNCERTAIN_WRITE_META_KEY,),
            ).fetchone()
        except sqlite3.Error:
            return False
        return row is not None and row["value"] == "1"

    @staticmethod
    def _create_schema_v1(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE gca_host_store_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """,
        )
        conn.execute(
            """
            CREATE TABLE gca_execution_results (
                execution_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            """,
        )
        conn.execute(
            """
            CREATE TABLE gca_execution_states (
                execution_id TEXT PRIMARY KEY,
                state_value TEXT NOT NULL
            );
            """,
        )
        conn.execute(
            """
            CREATE TABLE gca_execution_events (
                execution_id TEXT PRIMARY KEY,
                event_json TEXT NOT NULL
            );
            """,
        )
        conn.execute(
            """
            CREATE TABLE gca_create_follow_on (
                execution_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            """,
        )
        conn.execute(
            """
            CREATE TABLE gca_proof_receipts (
                execution_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            """,
        )
        conn.execute(
            """
            CREATE TABLE gca_policy_bundles (
                bundle_id TEXT NOT NULL,
                version TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (bundle_id, version)
            );
            """,
        )
        conn.execute(
            """
            CREATE TABLE gca_continuations (
                task_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            """,
        )
        conn.execute(
            "INSERT INTO gca_host_store_meta (key, value) VALUES (?, ?)",
            (_SCHEMA_META_KEY, _SCHEMA_VERSION),
        )

    def _bootstrap_schema(self) -> None:
        with self._managed_connection() as conn:
            if not self._had_existing_content:
                tables = self._list_user_tables(conn)
                if not tables:
                    self._create_schema_v1(conn)
                    conn.commit()
                    return
                raise GovernedExternalWorkHostPersistenceError(
                    "gca host store schema present in new database file",
                )

            row = conn.execute("PRAGMA integrity_check;").fetchone()
            if row is None or row[0] != "ok":
                raise GovernedExternalWorkHostPersistenceError(
                    "gca host sqlite integrity check failed",
                )

            tables = self._list_user_tables(conn)
            if tables != _REQUIRED_TABLES:
                raise GovernedExternalWorkHostPersistenceError(
                    "incomplete gca host store schema",
                )

            version = self._read_schema_version(conn)
            if version is None:
                raise GovernedExternalWorkHostPersistenceError(
                    "missing gca host store schema metadata",
                )
            if version != _SCHEMA_VERSION:
                raise GovernedExternalWorkHostPersistenceError(
                    f"unsupported gca host store schema:{version}",
                )

            if self._read_uncertain_write_flag(conn):
                self._writes_uncertain = True

    def _commit(self, conn: sqlite3.Connection) -> None:
        try:
            conn.commit()
        except sqlite3.Error as exc:
            try:
                conn.rollback()
            except sqlite3.Error:
                self._writes_uncertain = True
            raise GovernedExternalWorkHostPersistenceError(str(exc)) from exc


class _SqliteGovernedExecutionStore:
    __slots__ = ("_db",)

    def __init__(self, db: _SqliteGovernedExternalWorkHostDatabase) -> None:
        self._db = db

    @property
    def is_durable(self) -> bool:
        return True

    def put_result(self, result: AnyGovernedExecutionResult) -> None:
        key = str(result.execution_id)
        payload = canonical_json_text(result.model_dump(mode="json"))
        with self._db._managed_connection() as conn:
            conn.execute(
                """
                INSERT INTO gca_execution_results (execution_id, payload) VALUES (?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET payload = excluded.payload
                """,
                (key, payload),
            )
            self._db._commit(conn)

    def get_result(self, execution_id: str) -> AnyGovernedExecutionResult | None:
        with self._db._managed_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM gca_execution_results WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                return parse_governed_execution_result_json(row["payload"])
            except (ValueError, ValidationError) as exc:
                raise ValueError(f"corrupted_execution_artifact:{execution_id}") from exc

    def put_state(
        self,
        execution_id: str,
        state: GovernedExternalWorkHostState,
    ) -> None:
        with self._db._managed_connection() as conn:
            conn.execute(
                """
                INSERT INTO gca_execution_states (execution_id, state_value) VALUES (?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET state_value = excluded.state_value
                """,
                (execution_id, state.value),
            )
            self._db._commit(conn)

    def get_state(self, execution_id: str) -> GovernedExternalWorkHostState | None:
        with self._db._managed_connection() as conn:
            row = conn.execute(
                "SELECT state_value FROM gca_execution_states WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                return GovernedExternalWorkHostState(str(row["state_value"]).strip())
            except ValueError as exc:
                raise ValueError(f"corrupted_execution_state:{execution_id}") from exc

    def put_event_json(self, execution_id: str, event_json: str) -> None:
        with self._db._managed_connection() as conn:
            conn.execute(
                """
                INSERT INTO gca_execution_events (execution_id, event_json) VALUES (?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET event_json = excluded.event_json
                """,
                (execution_id, event_json),
            )
            self._db._commit(conn)

    def get_event_json(self, execution_id: str) -> str | None:
        with self._db._managed_connection() as conn:
            row = conn.execute(
                "SELECT event_json FROM gca_execution_events WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
            if row is None:
                return None
            return str(row["event_json"])

    def put_create_follow_on_record(
        self,
        execution_id: str,
        record: GovernedExternalWorkCreateFollowOnRecordV1,
    ) -> None:
        key = execution_id.strip()
        if str(record.execution_id) != key:
            raise ValueError("follow_on_execution_id_key_mismatch")
        payload = canonical_json_text(record.model_dump(mode="json"))
        with self._db._managed_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM gca_create_follow_on WHERE execution_id = ?",
                (key,),
            ).fetchone()
            if row is not None:
                existing = parse_governed_external_work_create_follow_on_record_json(
                    row["payload"],
                )
                if follow_on_records_equivalent(existing, record):
                    return
                raise GovernedExecutionCreateFollowOnConflictError(
                    f"conflicting_create_follow_on_record:{key}",
                )
            conn.execute(
                "INSERT INTO gca_create_follow_on (execution_id, payload) VALUES (?, ?)",
                (key, payload),
            )
            self._db._commit(conn)

    def get_create_follow_on_record(
        self,
        execution_id: str,
    ) -> GovernedExternalWorkCreateFollowOnRecordV1 | None:
        with self._db._managed_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM gca_create_follow_on WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                return parse_governed_external_work_create_follow_on_record_json(
                    row["payload"],
                )
            except ValueError as exc:
                raise ValueError(
                    f"corrupted_create_follow_on_artifact:{execution_id}",
                ) from exc


class _SqliteProofReceiptStore:
    __slots__ = ("_db",)

    def __init__(self, db: _SqliteGovernedExternalWorkHostDatabase) -> None:
        self._db = db

    @property
    def is_durable(self) -> bool:
        return True

    def put_receipt(self, execution_id: str, receipt: ExecutionEvidenceProofReceipt) -> None:
        payload = canonical_json_text(receipt.model_dump(mode="json"))
        with self._db._managed_connection() as conn:
            conn.execute(
                """
                INSERT INTO gca_proof_receipts (execution_id, payload) VALUES (?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET payload = excluded.payload
                """,
                (execution_id, payload),
            )
            self._db._commit(conn)

    def get_receipt(self, execution_id: str) -> ExecutionEvidenceProofReceipt | None:
        with self._db._managed_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM gca_proof_receipts WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                return parse_execution_evidence_proof_receipt_json(row["payload"])
            except (ValueError, json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"corrupted_receipt_artifact:{execution_id}") from exc


class _SqlitePolicyBundleArtifactStore:
    __slots__ = ("_db",)

    def __init__(self, db: _SqliteGovernedExternalWorkHostDatabase) -> None:
        self._db = db

    @property
    def is_durable(self) -> bool:
        return True

    def put_bundle(self, bundle: ImmutableRuntimePolicyBundle) -> None:
        payload = canonical_json_text(bundle.model_dump(mode="json"))
        with self._db._managed_connection() as conn:
            conn.execute(
                """
                INSERT INTO gca_policy_bundles (bundle_id, version, payload) VALUES (?, ?, ?)
                ON CONFLICT(bundle_id, version) DO UPDATE SET payload = excluded.payload
                """,
                (bundle.bundle_id, bundle.version, payload),
            )
            self._db._commit(conn)

    def get_bundle(
        self,
        bundle_id: str,
        version: str,
    ) -> ImmutableRuntimePolicyBundle | None:
        with self._db._managed_connection() as conn:
            row = conn.execute(
                """
                SELECT payload FROM gca_policy_bundles
                WHERE bundle_id = ? AND version = ?
                """,
                (bundle_id, version),
            ).fetchone()
            if row is None:
                return None
            try:
                return ImmutableRuntimePolicyBundle.model_validate_json(row["payload"])
            except ValidationError as exc:
                raise ValueError(
                    f"corrupted_bundle_artifact:{bundle_id}:{version}",
                ) from exc


class _SqliteContinuationStateStore:
    __slots__ = ("_db",)

    def __init__(self, db: _SqliteGovernedExternalWorkHostDatabase) -> None:
        self._db = db

    @property
    def is_durable(self) -> bool:
        return True

    def put_continuation(
        self,
        task_id: str,
        continuation: GovernedContinuationRequest,
    ) -> None:
        payload = canonical_json_text(continuation.model_dump(mode="json"))
        with self._db._managed_connection() as conn:
            conn.execute(
                """
                INSERT INTO gca_continuations (task_id, payload) VALUES (?, ?)
                ON CONFLICT(task_id) DO UPDATE SET payload = excluded.payload
                """,
                (task_id, payload),
            )
            self._db._commit(conn)

    def get_continuation(self, task_id: str) -> GovernedContinuationRequest | None:
        with self._db._managed_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM gca_continuations WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                return GovernedContinuationRequest.model_validate_json(row["payload"])
            except ValidationError as exc:
                raise ValueError(f"corrupted_continuation_artifact:{task_id}") from exc


class SQLiteGovernedExternalWorkHostStoreBackend(OwnedGovernedExternalWorkHostStoreBundle):
    """One SQLite database backing all four GCA host store protocols."""

    __slots__ = ("_bundle", "_db")

    def __init__(self, db_path: Path) -> None:
        self._db = _SqliteGovernedExternalWorkHostDatabase(db_path)
        self._bundle = GovernedExternalWorkHostStoreBundle(
            execution_store=_SqliteGovernedExecutionStore(self._db),
            receipt_store=_SqliteProofReceiptStore(self._db),
            bundle_store=_SqlitePolicyBundleArtifactStore(self._db),
            continuation_store=_SqliteContinuationStateStore(self._db),
        )

    @property
    def host_store_bundle(self) -> GovernedExternalWorkHostStoreBundle:
        return self._bundle

    @property
    def bundle(self) -> GovernedExternalWorkHostStoreBundle:
        return self._bundle

    @property
    def db_path(self) -> Path:
        return self._db.db_path

    def close(self) -> None:
        self._db.close()


def open_sqlite_governed_external_work_host_store_bundle_at(
    db_path: Path,
) -> SQLiteGovernedExternalWorkHostStoreBackend:
    """Open a path-backed durable GCA host store bundle (explicit trusted path required)."""
    return SQLiteGovernedExternalWorkHostStoreBackend(db_path)


__all__ = [
    "GovernedExternalWorkHostPersistenceError",
    "SQLiteGovernedExternalWorkHostStoreBackend",
    "open_sqlite_governed_external_work_host_store_bundle_at",
]
