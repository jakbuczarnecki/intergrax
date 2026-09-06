# © Artur Czarnecki. All rights reserved.

"""SQLite reference durable adapter for ``DistributedKVStore``."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from intergrax.distributed.contracts.kv_store import DistributedKVStore


class SqliteDistributedKVStore(DistributedKVStore):
    """Process-safe SQLite-backed tenant-scoped key-value store."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS distributed_kv_entries (
                        tenant_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value BLOB NOT NULL,
                        PRIMARY KEY (tenant_id, key)
                    )
                    """
                )

    def get(self, tenant_id: str, key: str) -> Optional[bytes]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value FROM distributed_kv_entries WHERE tenant_id = ? AND key = ?",
                    (tenant_id, key),
                ).fetchone()
                if row is None:
                    return None
                return bytes(row[0])

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        del ttl_seconds
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO distributed_kv_entries (tenant_id, key, value)
                    VALUES (?, ?, ?)
                    ON CONFLICT(tenant_id, key) DO UPDATE SET value = excluded.value
                    """,
                    (tenant_id, key, value),
                )

    def delete(self, tenant_id: str, key: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM distributed_kv_entries WHERE tenant_id = ? AND key = ?",
                    (tenant_id, key),
                )

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: Optional[bytes],
        new_value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        del ttl_seconds
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value FROM distributed_kv_entries WHERE tenant_id = ? AND key = ?",
                    (tenant_id, key),
                ).fetchone()
                current = bytes(row[0]) if row is not None else None
                if expected is None and current is not None:
                    return False
                if expected is not None and current != expected:
                    return False
                conn.execute(
                    """
                    INSERT INTO distributed_kv_entries (tenant_id, key, value)
                    VALUES (?, ?, ?)
                    ON CONFLICT(tenant_id, key) DO UPDATE SET value = excluded.value
                    """,
                    (tenant_id, key, new_value),
                )
                return True


def build_sqlite_distributed_kv_store(db_path: Path) -> SqliteDistributedKVStore:
    """Construct one durable process-local ``DistributedKVStore`` universe."""
    return SqliteDistributedKVStore(db_path)


__all__ = [
    "SqliteDistributedKVStore",
    "build_sqlite_distributed_kv_store",
]
