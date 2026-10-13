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
_SCHEMA_VERSION = "1"


def _encode_invocation(invocation: ProviderInvocation) -> str:
    return canonical_json_text(invocation.model_dump(mode="json"))


def _decode_invocation(payload: str) -> ProviderInvocation:
    data = json.loads(payload)
    if data.get("schema_version") != SCHEMA_PROVIDER_INVOCATION_V1:
        raise ProviderInvocationPersistenceError("invalid provider invocation schema in store")
    return ProviderInvocation.model_validate(data)


def _encode_outcome(outcome: ProviderInvocationOutcome) -> str:
    return canonical_json_text(outcome.model_dump(mode="json"))


def _decode_outcome(payload: str) -> ProviderInvocationOutcome:
    data = json.loads(payload)
    if data.get("schema_version") != SCHEMA_PROVIDER_INVOCATION_OUTCOME_V1:
        raise ProviderInvocationPersistenceError("invalid provider invocation outcome schema in store")
    return ProviderInvocationOutcome.model_validate(data)


class SQLiteProviderInvocationStore(ProviderInvocationStore):
    """Process-local durable invocation intent/outcome store (WAL, synchronous=FULL)."""

    __slots__ = ("_closed", "_db_path", "_lock")

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._closed = False
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
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

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise ProviderInvocationPersistenceError("provider invocation store is closed")
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=FULL;")
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            raise ProviderInvocationPersistenceError(str(exc)) from exc
        return conn

    @contextmanager
    def _managed_connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connect()
            try:
                yield conn
            finally:
                conn.close()

    def _bootstrap_schema(self) -> None:
        with self._managed_connection() as conn:
            if self._db_path.exists() and self._db_path.stat().st_size > 0:
                row = conn.execute("PRAGMA integrity_check;").fetchone()
                if row is None or row[0] != "ok":
                    raise ProviderInvocationPersistenceError(
                        "provider invocation sqlite integrity check failed",
                    )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_invocation_store_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """,
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_invocation_intents (
                    invocation_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                """,
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_invocation_outcomes (
                    invocation_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                """,
            )
            meta = conn.execute(
                "SELECT value FROM provider_invocation_store_meta WHERE key = ?",
                (_SCHEMA_META_KEY,),
            ).fetchone()
            if meta is None:
                conn.execute(
                    "INSERT INTO provider_invocation_store_meta (key, value) VALUES (?, ?)",
                    (_SCHEMA_META_KEY, _SCHEMA_VERSION),
                )
            elif meta["value"] != _SCHEMA_VERSION:
                raise ProviderInvocationPersistenceError(
                    f"unsupported provider invocation store schema:{meta['value']}",
                )
            conn.commit()

    @staticmethod
    def _commit_transaction(conn: sqlite3.Connection) -> None:
        try:
            conn.commit()
        except sqlite3.Error as exc:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            raise ProviderInvocationPersistenceError(str(exc)) from exc

    def insert_invocation_if_absent(
        self,
        invocation: ProviderInvocation,
    ) -> ProviderInvocationIntentWriteResult:
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
                        conn.rollback()
                        return ProviderInvocationIntentWriteResult.EQUIVALENT_NOOP
                    conn.rollback()
                    raise ProviderInvocationConflictError(
                        f"provider invocation conflict:{invocation.invocation_id}",
                    )
                conn.execute(
                    "INSERT INTO provider_invocation_intents (invocation_id, payload) VALUES (?, ?)",
                    (invocation.invocation_id, _encode_invocation(invocation)),
                )
                self._commit_transaction(conn)
                return ProviderInvocationIntentWriteResult.INSERTED
            except ProviderInvocationConflictError:
                raise
            except ProviderInvocationPersistenceError:
                raise
            except sqlite3.Error as exc:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise ProviderInvocationPersistenceError(str(exc)) from exc

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
                        conn.rollback()
                        return
                    conn.rollback()
                    raise ProviderInvocationOutcomeConflictError(
                        f"provider invocation outcome conflict:{outcome.invocation_id}",
                    )
                conn.execute(
                    "INSERT INTO provider_invocation_outcomes (invocation_id, payload) VALUES (?, ?)",
                    (outcome.invocation_id, _encode_outcome(outcome)),
                )
                self._commit_transaction(conn)
            except (
                ProviderInvocationOutcomeConflictError,
                ProviderInvocationPersistenceError,
            ):
                raise
            except sqlite3.Error as exc:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise ProviderInvocationPersistenceError(str(exc)) from exc

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
