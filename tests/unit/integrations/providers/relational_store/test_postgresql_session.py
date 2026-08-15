# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for PostgreSQL session/connection provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.postgresql.config import PostgreSQLIntegrationConfig
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    PostgreSQLIsolationLevel,
)

pytestmark = pytest.mark.unit


class _FakeCursor:
    def __init__(self, rows: list[dict[str, Any]] | None = None, rowcount: int = 0) -> None:
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self) -> dict[str, Any] | None:
        if not self._rows:
            return None
        return self._rows[0]

    def fetchall(self) -> list[dict[str, Any]]:
        return self._rows


class _FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[Any, Sequence[Any]]] = []
        self.committed = 0
        self.rolled_back = 0
        self.closed = False

    def execute(self, sql: Any, params: Sequence[Any] = ()) -> _FakeCursor:
        self.executed.append((sql, params))
        return _FakeCursor([{"value": 1}], rowcount=1)

    def commit(self) -> None:
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1

    def close(self) -> None:
        self.closed = True


def _connection_factory() -> tuple[_FakeConnection, callable]:
    conn = _FakeConnection()

    def factory() -> _FakeConnection:
        return conn

    return conn, factory


def test_provider_transaction_commits_on_success() -> None:
    conn, factory = _connection_factory()
    provider = PostgreSQLConnectionProvider(
        PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test", tenant_schema="tenant_a"),
        connection_factory=factory,
    )

    with provider.transaction(isolation_level=PostgreSQLIsolationLevel.READ_COMMITTED) as session:
        result = session.execute("SELECT 1")
        assert result.fetchone() == {"value": 1}

    assert conn.committed == 1
    assert conn.rolled_back == 0
    assert conn.closed is True
    assert any(
        "SET TRANSACTION ISOLATION LEVEL READ COMMITTED" in str(item[0])
        for item in conn.executed
    )


def test_provider_transaction_rollbacks_on_exception() -> None:
    conn, factory = _connection_factory()
    provider = PostgreSQLConnectionProvider(
        PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test"),
        connection_factory=factory,
    )

    with pytest.raises(RuntimeError, match="domain failure"):
        with provider.transaction() as session:
            session.execute("SELECT 1")
            raise RuntimeError("domain failure")

    assert conn.rolled_back == 1
    assert conn.closed is True


def test_provider_rejects_invalid_schema_before_execution() -> None:
    with pytest.raises(ValueError, match="tenant_schema must be a simple SQL identifier"):
        PostgreSQLConnectionProvider(
            PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test"),
            tenant_schema="bad-schema",
        )


def test_provider_open_configured_connection_applies_search_path() -> None:
    conn, factory = _connection_factory()
    provider = PostgreSQLConnectionProvider(
        PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test", tenant_schema="tenant_a"),
        connection_factory=factory,
    )
    raw = provider.open_configured_connection()
    assert raw is conn
    assert conn.committed == 1
    assert len(conn.executed) >= 1


def test_missing_psycopg_raises_configuration_error() -> None:
    from intergrax.integrations.providers.relational_store.postgresql import session as session_module

    with patch.object(
        session_module,
        "import_psycopg",
        side_effect=IntegrationConfigurationError("missing psycopg"),
    ):
        with pytest.raises(IntegrationConfigurationError, match="missing psycopg"):
            session_module.import_psycopg()


def test_provider_uses_psycopg_connect_when_no_connection_factory() -> None:
    config = PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test", tenant_schema="tenant_a")
    mock_conn = MagicMock()
    mock_psycopg = MagicMock()
    mock_psycopg.connect.return_value = mock_conn
    dict_row = MagicMock()

    with patch(
        "intergrax.integrations.providers.relational_store.postgresql.session.import_psycopg",
        return_value=(mock_psycopg, MagicMock(), dict_row, MagicMock()),
    ):
        provider = PostgreSQLConnectionProvider(config)
        provider.open_configured_connection()

    mock_psycopg.connect.assert_called_once_with(
        "postgresql://localhost/test",
        row_factory=dict_row,
    )
    assert mock_conn.execute.called
    assert mock_conn.commit.call_count == 1


def test_session_execute_statement_delegates_to_connection() -> None:
    conn, factory = _connection_factory()
    provider = PostgreSQLConnectionProvider(
        PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test", tenant_schema="tenant_a"),
        connection_factory=factory,
    )
    statement = object()

    with provider.connection() as session:
        session.execute_statement(statement)

    assert conn.executed[-1] == (statement, ())


def test_ensure_schema_exists_uses_session_public_api() -> None:
    conn, factory = _connection_factory()
    provider = PostgreSQLConnectionProvider(
        PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test", tenant_schema="tenant_a"),
        connection_factory=factory,
    )

    with provider.connection() as session:
        provider.ensure_schema_exists(session, "tenant_a")

    assert len(conn.executed) == 2
    assert "CREATE SCHEMA" in str(conn.executed[-1][0])


def test_ensure_schema_exists_skips_public_schema() -> None:
    conn, factory = _connection_factory()
    provider = PostgreSQLConnectionProvider(
        PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test", tenant_schema="public"),
        connection_factory=factory,
    )

    with provider.connection() as session:
        provider.ensure_schema_exists(session, "public")

    assert len(conn.executed) == 1
    assert "search_path" in str(conn.executed[0][0])
