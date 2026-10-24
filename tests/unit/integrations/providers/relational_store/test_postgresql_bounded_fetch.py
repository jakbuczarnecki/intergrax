# © Artur Czarnecki. All rights reserved.

"""L1 unit tests for PostgreSQL bounded query physical fetch path."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import pytest
from psycopg import pq

from intergrax.integrations.providers.relational_store.postgresql.adapter import (
    _PostgreSQLRelationalStore,
)
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)

pytestmark = pytest.mark.unit


class _ConnectionInfo:
    def __init__(self, status: pq.TransactionStatus) -> None:
        self.transaction_status = status


class _IterableCursor:
    def __init__(self, rows: list[dict[str, Any]], *, fail_on_iter: bool = False) -> None:
        self._rows = rows
        self._fail_on_iter = fail_on_iter
        self.itersize = 0
        self.closed = False
        self.executed: tuple[str, Sequence[Any]] | None = None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self.executed = (sql, params)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._fail_on_iter:
            raise RuntimeError("bounded fetch failed mid-stream")
        return iter(self._rows)

    def close(self) -> None:
        self.closed = True


class _BoundedFakeConnection:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        transaction_status: pq.TransactionStatus = pq.TransactionStatus.IDLE,
    ) -> None:
        self._rows = rows
        self.executed: list[str] = []
        self.committed = 0
        self.rolled_back = 0
        self.cursor_names: list[str] = []
        self._info = _ConnectionInfo(transaction_status)

    @property
    def info(self) -> _ConnectionInfo:
        return self._info

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
    assert conn.rolled_back == 0


def test_fetch_query_bounded_inside_caller_transaction_does_not_commit_or_begin() -> None:
    conn = _BoundedFakeConnection(
        [{"id": 1}],
        transaction_status=pq.TransactionStatus.INTRANS,
    )
    store = _PostgreSQLRelationalStore(
        PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test"),
        conn,
    )

    rows, truncated, reason = store.fetch_query_bounded(
        "SELECT id FROM t",
        max_rows=5,
        max_result_bytes=10_000,
    )

    assert rows == ({"id": 1},)
    assert truncated is False
    assert reason == ""
    assert "BEGIN READ ONLY" not in conn.executed
    assert conn.committed == 0
    assert conn.rolled_back == 0


def test_fetch_query_bounded_failure_rolls_back_only_owned_transaction() -> None:
    class _FailingConnection(_BoundedFakeConnection):
        def cursor(self, name: str | None = None, *args: object, **kwargs: object) -> _IterableCursor:
            assert name is not None
            self.cursor_names.append(name)
            return _IterableCursor(self._rows, fail_on_iter=True)

    failing = _FailingConnection([], transaction_status=pq.TransactionStatus.IDLE)
    store_fail = _PostgreSQLRelationalStore(
        PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test"),
        failing,
    )
    with pytest.raises(RuntimeError, match="mid-stream"):
        store_fail.fetch_query_bounded("SELECT 1", max_rows=1, max_result_bytes=100)
    assert failing.committed == 0
    assert failing.rolled_back == 1

    caller_conn = _BoundedFakeConnection(
        [],
        transaction_status=pq.TransactionStatus.INTRANS,
    )
    caller_store = _PostgreSQLRelationalStore(
        PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test"),
        caller_conn,
    )

    class _CallerFailingConnection(_BoundedFakeConnection):
        def cursor(self, name: str | None = None, *args: object, **kwargs: object) -> _IterableCursor:
            assert name is not None
            self.cursor_names.append(name)
            return _IterableCursor(self._rows, fail_on_iter=True)

    caller_failing = _CallerFailingConnection([], transaction_status=pq.TransactionStatus.INTRANS)
    caller_store_fail = _PostgreSQLRelationalStore(
        PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test"),
        caller_failing,
    )
    with pytest.raises(RuntimeError, match="mid-stream"):
        caller_store_fail.fetch_query_bounded("SELECT 1", max_rows=1, max_result_bytes=100)
    assert caller_failing.rolled_back == 0
