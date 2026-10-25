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
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLBoundedFetchSafetyError,
    PostgreSQLBoundedFetchTransactionScope,
    classify_postgresql_bounded_fetch_transaction_scope,
)

pytestmark = pytest.mark.unit


class _ConnectionInfo:
    def __init__(self, status: pq.TransactionStatus) -> None:
        self.transaction_status = status


class _IterableCursor:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        fail_on_iter: bool = False,
        fail_on_close: bool = False,
    ) -> None:
        self._rows = rows
        self._fail_on_iter = fail_on_iter
        self._fail_on_close = fail_on_close
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
        if self._fail_on_close:
            raise RuntimeError("cursor close failed")
        self.closed = True


class _BoundedFakeConnection:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        transaction_status: pq.TransactionStatus = pq.TransactionStatus.IDLE,
        closed: bool = False,
        fail_commit: bool = False,
        fail_rollback: bool = False,
    ) -> None:
        self._rows = rows
        self.executed: list[str] = []
        self.committed = 0
        self.rolled_back = 0
        self.cursor_names: list[str] = []
        self._info = _ConnectionInfo(transaction_status)
        self.closed = closed
        self._fail_commit = fail_commit
        self._fail_rollback = fail_rollback
        self.lifecycle_events: list[str] = []
        self._cursor_fail_on_close = False
        self._cursor_fail_on_iter = False

    @property
    def info(self) -> _ConnectionInfo:
        return self._info

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self.executed.append(sql)

    def cursor(self, name: str | None = None, *args: object, **kwargs: object) -> _IterableCursor:
        assert name is not None
        self.cursor_names.append(name)
        return _IterableCursor(
            self._rows,
            fail_on_iter=self._cursor_fail_on_iter,
            fail_on_close=self._cursor_fail_on_close,
        )

    def commit(self) -> None:
        self.lifecycle_events.append("commit")
        if self._fail_commit:
            raise RuntimeError("commit failed")
        self.committed += 1

    def rollback(self) -> None:
        self.lifecycle_events.append("rollback")
        if self._fail_rollback:
            raise RuntimeError("rollback failed")
        self.rolled_back += 1


def _store(conn: _BoundedFakeConnection) -> _PostgreSQLRelationalStore:
    return _PostgreSQLRelationalStore(
        PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test"),
        conn,
    )


@pytest.mark.parametrize(
    ("status", "expected_scope"),
    [
        (pq.TransactionStatus.IDLE, PostgreSQLBoundedFetchTransactionScope.ADAPTER_OWNED_READONLY),
        (pq.TransactionStatus.INTRANS, PostgreSQLBoundedFetchTransactionScope.CALLER_OWNED),
    ],
)
def test_classify_scope_idle_and_intrans(
    status: pq.TransactionStatus,
    expected_scope: PostgreSQLBoundedFetchTransactionScope,
) -> None:
    conn = _BoundedFakeConnection([], transaction_status=status)
    assert classify_postgresql_bounded_fetch_transaction_scope(conn) == expected_scope


@pytest.mark.parametrize(
    ("status", "label"),
    [
        (pq.TransactionStatus.INERROR, "INERROR"),
        (pq.TransactionStatus.ACTIVE, "ACTIVE"),
        (pq.TransactionStatus.UNKNOWN, "UNKNOWN"),
    ],
)
def test_classify_scope_rejects_unsafe_states(
    status: pq.TransactionStatus,
    label: str,
) -> None:
    conn = _BoundedFakeConnection([], transaction_status=status)
    with pytest.raises(PostgreSQLBoundedFetchSafetyError) as exc_info:
        classify_postgresql_bounded_fetch_transaction_scope(conn)
    assert exc_info.value.transaction_status == label


def test_classify_scope_rejects_closed_connection() -> None:
    conn = _BoundedFakeConnection([], closed=True)
    with pytest.raises(PostgreSQLBoundedFetchSafetyError, match="closed"):
        classify_postgresql_bounded_fetch_transaction_scope(conn)


def test_fetch_query_bounded_opens_read_only_transaction_and_named_cursor() -> None:
    conn = _BoundedFakeConnection([{"id": 1}, {"id": 2}, {"id": 3}])
    store = _store(conn)

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
    assert conn.lifecycle_events == ["commit"]


def test_fetch_query_bounded_closes_cursor_before_commit() -> None:
    conn = _BoundedFakeConnection([{"id": 1}])
    store = _store(conn)
    store.fetch_query_bounded("SELECT 1", max_rows=5, max_result_bytes=10_000)
    assert conn.lifecycle_events == ["commit"]


def test_fetch_query_bounded_inside_caller_transaction_does_not_commit_or_begin() -> None:
    conn = _BoundedFakeConnection(
        [{"id": 1}],
        transaction_status=pq.TransactionStatus.INTRANS,
    )
    store = _store(conn)

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
    assert conn.lifecycle_events == []


def test_fetch_query_bounded_failure_rolls_back_only_owned_transaction() -> None:
    class _FailingConnection(_BoundedFakeConnection):
        def cursor(self, name: str | None = None, *args: object, **kwargs: object) -> _IterableCursor:
            assert name is not None
            self.cursor_names.append(name)
            return _IterableCursor(self._rows, fail_on_iter=True)

    failing = _FailingConnection([], transaction_status=pq.TransactionStatus.IDLE)
    store_fail = _store(failing)
    with pytest.raises(RuntimeError, match="mid-stream"):
        store_fail.fetch_query_bounded("SELECT 1", max_rows=1, max_result_bytes=100)
    assert failing.committed == 0
    assert failing.rolled_back == 1

    caller_failing = _FailingConnection([], transaction_status=pq.TransactionStatus.INTRANS)
    caller_store_fail = _store(caller_failing)
    with pytest.raises(RuntimeError, match="mid-stream"):
        caller_store_fail.fetch_query_bounded("SELECT 1", max_rows=1, max_result_bytes=100)
    assert caller_failing.rolled_back == 0


def test_fetch_query_bounded_cursor_close_error_surfaces_when_fetch_succeeded() -> None:
    conn = _BoundedFakeConnection([{"id": 1}])
    conn._cursor_fail_on_close = True
    store = _store(conn)
    with pytest.raises(RuntimeError, match="cursor close failed"):
        store.fetch_query_bounded("SELECT 1", max_rows=1, max_result_bytes=100)
    assert conn.committed == 0
    assert conn.rolled_back == 1


def test_fetch_query_bounded_preserves_fetch_error_over_cursor_close_error() -> None:
    conn = _BoundedFakeConnection([])
    conn._cursor_fail_on_iter = True
    conn._cursor_fail_on_close = True
    store = _store(conn)
    with pytest.raises(RuntimeError, match="mid-stream"):
        store.fetch_query_bounded("SELECT 1", max_rows=1, max_result_bytes=100)
    assert conn.rolled_back == 1


def test_fetch_query_bounded_commit_error_after_successful_fetch() -> None:
    conn = _BoundedFakeConnection([{"id": 1}], fail_commit=True)
    store = _store(conn)
    with pytest.raises(RuntimeError, match="commit failed"):
        store.fetch_query_bounded("SELECT 1", max_rows=1, max_result_bytes=100)
    assert conn.rolled_back == 1


@pytest.mark.parametrize(
    "status",
    [
        pq.TransactionStatus.INERROR,
        pq.TransactionStatus.ACTIVE,
        pq.TransactionStatus.UNKNOWN,
    ],
)
def test_fetch_query_bounded_rejects_unsafe_connection_state(
    status: pq.TransactionStatus,
) -> None:
    conn = _BoundedFakeConnection([], transaction_status=status)
    store = _store(conn)
    with pytest.raises(PostgreSQLBoundedFetchSafetyError):
        store.fetch_query_bounded("SELECT 1", max_rows=1, max_result_bytes=100)
    assert "BEGIN READ ONLY" not in conn.executed
    assert conn.committed == 0
    assert conn.rolled_back == 0


def test_fetch_query_bounded_rejects_closed_connection() -> None:
    conn = _BoundedFakeConnection([], closed=True)
    store = _store(conn)
    with pytest.raises(PostgreSQLBoundedFetchSafetyError, match="closed"):
        store.fetch_query_bounded("SELECT 1", max_rows=1, max_result_bytes=100)


def test_fetch_query_bounded_empty_result_adapter_owned() -> None:
    conn = _BoundedFakeConnection([])
    store = _store(conn)
    rows, truncated, reason = store.fetch_query_bounded(
        "SELECT 1 WHERE false",
        max_rows=5,
        max_result_bytes=500,
    )
    assert rows == ()
    assert truncated is False
    assert reason == ""
    assert conn.committed == 1
