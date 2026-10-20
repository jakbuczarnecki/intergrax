# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Dedicated SQLite ``ProviderInvocationStore`` (PROD-Q-R1-PIS-01 / ADR-03)."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    SCHEMA_PROVIDER_INVOCATION_OUTCOME_V1,
    SCHEMA_PROVIDER_INVOCATION_V1,
)
from intergrax.contracts.provider_invocation_store import (
    ProviderInvocationConflictError,
    ProviderInvocationIntentWriteResult,
    ProviderInvocationOutcomeConflictError,
    ProviderInvocationPersistenceError,
    ProviderInvocationStore,
    provider_invocation_outcomes_equivalent,
    provider_invocations_equivalent,
)
from intergrax.runtime.attestation.canonical_json import canonical_json_text

_SCHEMA_META_KEY = "schema_version"
_UNCERTAIN_WRITE_META_KEY = "durable_write_uncertain"
_SCHEMA_VERSION = "1"

_REQUIRED_TABLES = frozenset(
    {
        "provider_invocation_store_meta",
        "provider_invocation_intents",
        "provider_invocation_outcomes",
    },
)


def _encode_invocation(invocation: ProviderInvocation) -> str:
    return canonical_json_text(invocation.model_dump(mode="json"))


def _decode_invocation(payload: str) -> ProviderInvocation:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProviderInvocationPersistenceError(
            "corrupt provider invocation payload in store",
        ) from exc
    if not isinstance(data, dict):
        raise ProviderInvocationPersistenceError(
            "corrupt provider invocation payload in store",
        )
    if data.get("schema_version") != SCHEMA_PROVIDER_INVOCATION_V1:
        raise ProviderInvocationPersistenceError("invalid provider invocation schema in store")
    try:
        return ProviderInvocation.model_validate(data)
    except ValidationError as exc:
        raise ProviderInvocationPersistenceError(
            "invalid provider invocation payload in store",
        ) from exc


def _encode_outcome(outcome: ProviderInvocationOutcome) -> str:
    return canonical_json_text(outcome.model_dump(mode="json"))


def _decode_outcome(payload: str) -> ProviderInvocationOutcome:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProviderInvocationPersistenceError(
            "corrupt provider invocation outcome payload in store",
        ) from exc
    if not isinstance(data, dict):
        raise ProviderInvocationPersistenceError(
            "corrupt provider invocation outcome payload in store",
        )
    if data.get("schema_version") != SCHEMA_PROVIDER_INVOCATION_OUTCOME_V1:
        raise ProviderInvocationPersistenceError(
            "invalid provider invocation outcome schema in store",
        )
    try:
        return ProviderInvocationOutcome.model_validate(data)
    except ValidationError as exc:
        raise ProviderInvocationPersistenceError(
            "invalid provider invocation outcome payload in store",
        ) from exc


class SQLiteProviderInvocationStore(ProviderInvocationStore):
    """Process-local durable invocation intent/outcome store (WAL, synchronous=FULL)."""

    __slots__ = ("_closed", "_db_path", "_had_existing_content", "_lock", "_writes_uncertain")

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._closed = False
        self._writes_uncertain = False
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._had_existing_content = self._db_path.exists() and self._db_path.stat().st_size > 0
        self._bootstrap_schema()

    @property
    def is_durable(self) -> bool:
        return True

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
            raise ProviderInvocationPersistenceError("provider invocation store is closed")
        if self._writes_uncertain:
            raise ProviderInvocationPersistenceError(
                "provider invocation store transaction outcome uncertain",
            )

    def _connect(self) -> sqlite3.Connection:
        self._ensure_operational()
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=FULL;")
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            raise ProviderInvocationPersistenceError(str(exc)) from exc
        return conn

    def read_owned_connection_durability_pragmas(self) -> tuple[str, int]:
        """Non-authoritative qualification surface: pragmas on an application-owned connection."""
        with self._managed_connection() as conn:
            journal = str(conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower()
            synchronous = int(conn.execute("PRAGMA synchronous;").fetchone()[0])
        return journal, synchronous

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
                "SELECT value FROM provider_invocation_store_meta WHERE key = ?",
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
                "SELECT value FROM provider_invocation_store_meta WHERE key = ?",
                (_UNCERTAIN_WRITE_META_KEY,),
            ).fetchone()
        except sqlite3.Error:
            return False
        return row is not None and row["value"] == "1"

    @staticmethod
    def _create_schema_v1(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE provider_invocation_store_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """,
        )
        conn.execute(
            """
            CREATE TABLE provider_invocation_intents (
                invocation_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            """,
        )
        conn.execute(
            """
            CREATE TABLE provider_invocation_outcomes (
                invocation_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            """,
        )
        conn.execute(
            "INSERT INTO provider_invocation_store_meta (key, value) VALUES (?, ?)",
            (_SCHEMA_META_KEY, _SCHEMA_VERSION),
        )

    def _persist_uncertain_write_marker(self) -> None:
        """Best-effort separate transaction; not atomic with the failed write txn."""
        try:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO provider_invocation_store_meta (key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (_UNCERTAIN_WRITE_META_KEY, "1"),
                )
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise ProviderInvocationPersistenceError(
                "failed to persist durable provider invocation write-uncertainty marker",
            ) from exc

    def _bootstrap_schema(self) -> None:
        with self._managed_connection() as conn:
            if not self._had_existing_content:
                tables = self._list_user_tables(conn)
                if not tables:
                    self._create_schema_v1(conn)
                    conn.commit()
                    return
                raise ProviderInvocationPersistenceError(
                    "provider invocation store schema present in new database file",
                )

            row = conn.execute("PRAGMA integrity_check;").fetchone()
            if row is None or row[0] != "ok":
                raise ProviderInvocationPersistenceError(
                    "provider invocation sqlite integrity check failed",
                )

            tables = self._list_user_tables(conn)
            if tables != _REQUIRED_TABLES:
                raise ProviderInvocationPersistenceError(
                    "incomplete provider invocation store schema",
                )

            version = self._read_schema_version(conn)
            if version is None:
                raise ProviderInvocationPersistenceError(
                    "missing provider invocation store schema metadata",
                )
            if version != _SCHEMA_VERSION:
                raise ProviderInvocationPersistenceError(
                    f"unsupported provider invocation store schema:{version}",
                )

            if self._read_uncertain_write_flag(conn):
                self._writes_uncertain = True

    @staticmethod
    def _connection_commit(conn: sqlite3.Connection) -> None:
        conn.commit()

    @staticmethod
    def _connection_rollback(conn: sqlite3.Connection) -> None:
        conn.rollback()

    def _invocation_row_visible(
        self,
        conn: sqlite3.Connection,
        invocation_id: str,
    ) -> bool:
        row = conn.execute(
            "SELECT 1 FROM provider_invocation_intents WHERE invocation_id = ?",
            (invocation_id,),
        ).fetchone()
        return row is not None

    def _outcome_row_visible(
        self,
        conn: sqlite3.Connection,
        invocation_id: str,
    ) -> bool:
        row = conn.execute(
            "SELECT 1 FROM provider_invocation_outcomes WHERE invocation_id = ?",
            (invocation_id,),
        ).fetchone()
        return row is not None

    def _commit_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        pending_invocation_id: str | None = None,
        pending_outcome_id: str | None = None,
    ) -> None:
        try:
            self._connection_commit(conn)
        except sqlite3.Error as exc:
            rollback_ok = False
            try:
                self._connection_rollback(conn)
                rollback_ok = True
            except sqlite3.Error:
                rollback_ok = False

            uncertain = not rollback_ok
            if rollback_ok and pending_invocation_id is not None:
                if self._invocation_row_visible(conn, pending_invocation_id):
                    uncertain = True
            if rollback_ok and pending_outcome_id is not None:
                if self._outcome_row_visible(conn, pending_outcome_id):
                    uncertain = True

            if uncertain:
                self._writes_uncertain = True
            raise ProviderInvocationPersistenceError(str(exc)) from exc

    def _flush_uncertain_write_marker_if_needed(self) -> None:
        if not self._writes_uncertain:
            return
        self._persist_uncertain_write_marker()

    def insert_invocation_if_absent(
        self,
        invocation: ProviderInvocation,
    ) -> ProviderInvocationIntentWriteResult:
        try:
            with self._managed_connection() as conn:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT payload FROM provider_invocation_intents WHERE invocation_id = ?",
                        (invocation.invocation_id,),
                    ).fetchone()
                    if row is not None:
                        existing = _decode_invocation(row["payload"])
                        if provider_invocations_equivalent(existing, invocation):
                            self._connection_rollback(conn)
                            return ProviderInvocationIntentWriteResult.EQUIVALENT_NOOP
                        self._connection_rollback(conn)
                        raise ProviderInvocationConflictError(
                            f"provider invocation conflict:{invocation.invocation_id}",
                        )
                    conn.execute(
                        "INSERT INTO provider_invocation_intents (invocation_id, payload) VALUES (?, ?)",
                        (invocation.invocation_id, _encode_invocation(invocation)),
                    )
                    self._commit_transaction(
                        conn,
                        pending_invocation_id=invocation.invocation_id,
                    )
                    return ProviderInvocationIntentWriteResult.INSERTED
                except ProviderInvocationConflictError:
                    raise
                except ProviderInvocationPersistenceError:
                    raise
                except sqlite3.Error as exc:
                    try:
                        self._connection_rollback(conn)
                    except sqlite3.Error:
                        self._writes_uncertain = True
                    raise ProviderInvocationPersistenceError(str(exc)) from exc
        except ProviderInvocationPersistenceError:
            self._flush_uncertain_write_marker_if_needed()
            raise

    def put_invocation(self, invocation: ProviderInvocation) -> None:
        result = self.insert_invocation_if_absent(invocation)
        if result is ProviderInvocationIntentWriteResult.EQUIVALENT_NOOP:
            return
        if result is not ProviderInvocationIntentWriteResult.INSERTED:
            raise ProviderInvocationPersistenceError(
                "unexpected provider invocation insert result",
            )

    def get_invocation(self, invocation_id: str) -> ProviderInvocation | None:
        with self._managed_connection() as conn:
            try:
                row = conn.execute(
                    "SELECT payload FROM provider_invocation_intents WHERE invocation_id = ?",
                    (invocation_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise ProviderInvocationPersistenceError(str(exc)) from exc
            if row is None:
                return None
            return _decode_invocation(row["payload"])

    def put_outcome(self, outcome: ProviderInvocationOutcome) -> None:
        try:
            with self._managed_connection() as conn:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT payload FROM provider_invocation_outcomes WHERE invocation_id = ?",
                        (outcome.invocation_id,),
                    ).fetchone()
                    if row is not None:
                        existing = _decode_outcome(row["payload"])
                        if provider_invocation_outcomes_equivalent(existing, outcome):
                            self._connection_rollback(conn)
                            return
                        self._connection_rollback(conn)
                        raise ProviderInvocationOutcomeConflictError(
                            f"provider invocation outcome conflict:{outcome.invocation_id}",
                        )
                    conn.execute(
                        "INSERT INTO provider_invocation_outcomes (invocation_id, payload) VALUES (?, ?)",
                        (outcome.invocation_id, _encode_outcome(outcome)),
                    )
                    self._commit_transaction(
                        conn,
                        pending_outcome_id=outcome.invocation_id,
                    )
                except (
                    ProviderInvocationOutcomeConflictError,
                    ProviderInvocationPersistenceError,
                ):
                    raise
                except sqlite3.Error as exc:
                    try:
                        self._connection_rollback(conn)
                    except sqlite3.Error:
                        self._writes_uncertain = True
                    raise ProviderInvocationPersistenceError(str(exc)) from exc
        except ProviderInvocationPersistenceError:
            self._flush_uncertain_write_marker_if_needed()
            raise

    def get_outcome(self, invocation_id: str) -> ProviderInvocationOutcome | None:
        with self._managed_connection() as conn:
            try:
                row = conn.execute(
                    "SELECT payload FROM provider_invocation_outcomes WHERE invocation_id = ?",
                    (invocation_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise ProviderInvocationPersistenceError(str(exc)) from exc
            if row is None:
                return None
            return _decode_outcome(row["payload"])


def open_provider_invocation_store_at(db_path: Path) -> SQLiteProviderInvocationStore:
    """Open a path-backed durable ``ProviderInvocationStore`` (explicit path required)."""
    return SQLiteProviderInvocationStore(db_path)


__all__ = [
    "SQLiteProviderInvocationStore",
    "open_provider_invocation_store_at",
]
