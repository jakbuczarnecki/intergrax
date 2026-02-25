# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import base64
import pickle
import sqlite3
from typing import Optional
from pydantic import BaseModel

from intergrax.runtime.tools.idempotency_store import (
    IdempotencyStore,
    InvocationStatus,
)
from intergrax.tools.execution_models import ToolExecutionResult


class SQLiteIdempotencyStore(IdempotencyStore):
    """
    Persistent SQLite-backed ledger for tool idempotency.

    Guarantees:
    - tenant isolation via composite primary key
    - atomic STARTED via INSERT
    - crash-safe semantics
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_ledger (
                    tenant_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    PRIMARY KEY (tenant_id, key)
                )
                """
            )

    def get_status(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[InvocationStatus]:

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT status
                FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ?
                """,
                (tenant_id, key),
            ).fetchone()

        if row is None:
            return None

        return InvocationStatus(row[0])

    def record_started(
        self,
        tenant_id: str,
        key: str,
    ) -> None:

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO idempotency_ledger (tenant_id, key, status)
                    VALUES (?, ?, ?)
                    """,
                    (tenant_id, key, InvocationStatus.STARTED.value),
                )
        except sqlite3.IntegrityError:
            raise RuntimeError(
                f"Invocation already exists for key={key}"
            )

    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult[BaseModel],
    ) -> None:
        blob = base64.b64encode(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")

        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE idempotency_ledger
                SET status = ?, result_blob = ?
                WHERE tenant_id = ? AND key = ? AND status = ?
                """,
                (
                    InvocationStatus.COMPLETED.value,
                    blob,
                    tenant_id,
                    key,
                    InvocationStatus.STARTED.value,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("Invalid state transition to COMPLETED.")

    def get_completed_result(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[ToolExecutionResult[BaseModel]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT status, result_blob
                FROM idempotency_ledger
                WHERE tenant_id = ? AND key = ?
                """,
                (tenant_id, key),
            ).fetchone()

        if row is None:
            return None

        status, result_blob = row
        if status != InvocationStatus.COMPLETED.value:
            return None

        if result_blob is None:
            raise RuntimeError("Ledger inconsistency: COMPLETED without result_blob.")

        return pickle.loads(base64.b64decode(result_blob.encode("ascii")))