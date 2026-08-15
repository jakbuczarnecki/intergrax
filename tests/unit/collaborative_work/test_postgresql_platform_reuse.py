# © Artur Czarnecki. All rights reserved.

"""Architecture proofs that Collaborative Work reuses platform PostgreSQL infrastructure."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from intergrax.collaborative_work.postgresql_repository import PostgreSQLCollaborativeWorkStore
from intergrax.integrations.providers.relational_store.postgresql.config import PostgreSQLIntegrationConfig
from intergrax.integrations.providers.relational_store.postgresql.opens import open_postgresql_relational_store
from intergrax.integrations.providers.relational_store.postgresql.session import PostgreSQLConnectionProvider

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CW_POSTGRESQL_ADAPTER = _REPO_ROOT / "intergrax" / "collaborative_work" / "postgresql_repository.py"


def _import_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


class _FakeCursor:
    rowcount = 0

    def fetchone(self) -> None:
        return None

    def fetchall(self) -> list[Any]:
        return []


class _FakeConnection:
    def __init__(self) -> None:
        self.executed: list[Any] = []
        self.committed = 0
        self.closed = False

    def execute(self, sql: Any, params: tuple[Any, ...] = ()) -> _FakeCursor:
        self.executed.append(sql)
        return _FakeCursor()

    def commit(self) -> None:
        self.committed += 1

    def close(self) -> None:
        self.closed = True


def test_collaborative_work_store_uses_platform_connection_provider() -> None:
    conn = _FakeConnection()
    config = PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test", tenant_schema="tenant_a")

    with patch.object(PostgreSQLConnectionProvider, "ensure_schema_exists", return_value=None):
        store = PostgreSQLCollaborativeWorkStore(
            config,
            connection_factory=lambda: conn,
            schema_name="tenant_a",
        )

    assert isinstance(store._provider, PostgreSQLConnectionProvider)
    store.close()


def test_collaborative_work_adapter_does_not_import_psycopg() -> None:
    imported = _import_names(_CW_POSTGRESQL_ADAPTER)
    assert "psycopg" not in imported


def test_opens_and_collaborative_work_share_platform_provider() -> None:
    conn = _FakeConnection()
    config = PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test", tenant_schema="tenant_a")

    with patch.object(PostgreSQLConnectionProvider, "open_configured_connection", return_value=conn):
        open_postgresql_relational_store(config, connection_factory=lambda: conn)

    with patch.object(PostgreSQLConnectionProvider, "ensure_schema_exists", return_value=None):
        store = PostgreSQLCollaborativeWorkStore(
            config,
            connection_factory=lambda: conn,
            schema_name="tenant_a",
        )
    store.close()

    assert isinstance(store._provider, PostgreSQLConnectionProvider)


def test_invalid_schema_rejected_before_collaborative_work_store_open() -> None:
    config = PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test")
    with pytest.raises(ValueError, match="tenant_schema must be a simple SQL identifier"):
        PostgreSQLCollaborativeWorkStore(config, schema_name="bad-schema")
