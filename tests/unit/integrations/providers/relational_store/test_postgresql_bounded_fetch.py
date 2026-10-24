# © Artur Czarnecki. All rights reserved.

"""L1 unit tests for PostgreSQL bounded query physical fetch path."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import pytest

from intergrax.integrations.providers.relational_store.postgresql.adapter import (
    _PostgreSQLRelationalStore,
)
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)

pytestmark = pytest.mark.unit


class _IterableCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.itersize = 0
        self.closed = False
        self.executed: tuple[str, Sequence[Any]] | None = None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self.executed = (sql, params)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)

    def close(self) -> None:
        self.closed = True


class _BoundedFakeConnection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.executed: list[str] = []
        self.committed = 0
        self.rolled_back = 0
        self.cursor_names: list[str] = []

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self.executed.append(sql)

    def cursor(self, name: str | None = None, *args: object, **kwargs: object) -> _IterableCursor:
        assert name is not None
        self.cursor_names.append(name)
        return _IterableCursor(self._rows)

    def commit(self) -> None:
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1


def test_fetch_query_bounded_opens_read_only_transaction_and_named_cursor() -> None:
    conn = _BoundedFakeConnection([{"id": 1}, {"id": 2}, {"id": 3}])
    config = PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test")
    store = _PostgreSQLRelationalStore(config, conn)

    rows, truncated, reason = store.fetch_query_bounded(
        "SELECT id FROM t",
        max_rows=2,
        max_result_bytes=10_000,
    )

    assert rows == ({"id": 1}, {"id": 2})
    assert truncated is True
    assert reason == "row_limit"
    assert conn.executed[0] == "BEGIN READ ONLY"
    assert len(conn.cursor_names) == 1
    assert conn.cursor_names[0].startswith("igx_bq_")
    assert conn.committed == 1
