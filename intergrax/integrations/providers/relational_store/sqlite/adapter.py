# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite relational store adapter — generic SQL facade for ``RelationalStore``."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from intergrax.integrations.contracts.relational_store_bounded_query import (
    RelationalStoreBoundedQueryPort,
)
from intergrax.integrations.relational_bounded_row_accumulation import (
    accumulate_bounded_sql_rows,
)


class _SQLiteRelationalStore(RelationalStoreBoundedQueryPort):
    """
    Minimal ``RelationalStore`` over a single SQLite file.

    Instantiate via ``create_sqlite_relational_store()`` — not directly from app code.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None

    @property
    def db_path(self) -> Path:
        return self._db_path

    def connect(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        if self._connection is None:
            self._connection = sqlite3.connect(self._db_path)
            self._connection.row_factory = sqlite3.Row

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        conn = self._require_connection()
        conn.execute(sql, params)
        conn.commit()

    def fetch_all(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Sequence[Mapping[str, Any]]:
        conn = self._require_connection()
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def fetch_query_bounded(
        self,
        sql: str,
        params: Sequence[Any] = (),
        *,
        max_rows: int,
        max_result_bytes: int,
    ) -> tuple[tuple[Mapping[str, Any], ...], bool, str]:
        conn = self._require_connection()
        cursor = conn.execute(sql, params)
        try:

            def _row_iter():
                for row in cursor:
                    yield dict(row)

            accumulated, truncated, reason = accumulate_bounded_sql_rows(
                _row_iter(),
                max_rows=max_rows,
                max_result_bytes=max_result_bytes,
            )
            return tuple(accumulated), truncated, reason
        finally:
            cursor.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def release_connection_for_collaborative_work_materialization(self) -> str:
        """Relinquish generic connection before Collaborative Work opens its store."""
        self.close()
        return str(self._db_path)

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        return self._connection
