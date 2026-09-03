# © Artur Czarnecki. All rights reserved.

"""Architecture proofs that Autonomous Work reuses platform PostgreSQL infrastructure."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from intergrax.autonomous_work.postgresql_repository import PostgreSQLAutonomousWorkStore
from intergrax.integrations.providers.relational_store.postgresql.config import PostgreSQLIntegrationConfig
from intergrax.integrations.providers.relational_store.postgresql.opens import open_postgresql_relational_store
from intergrax.integrations.providers.relational_store.postgresql.session import PostgreSQLConnectionProvider

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AW_POSTGRESQL_ADAPTER = _REPO_ROOT / "intergrax" / "autonomous_work" / "postgresql_repository.py"


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

    def fetchone(self) -> dict[str, int]:
        return {"schema_version": 1}

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


def _open_store_with_injected_provider(
    config: PostgreSQLIntegrationConfig,
    conn: _FakeConnection,
    *,
    schema_name: str,
) -> tuple[PostgreSQLAutonomousWorkStore, PostgreSQLConnectionProvider]:
    provider = PostgreSQLConnectionProvider(
        config,
        connection_factory=lambda: conn,
        tenant_schema=schema_name,
    )
    with patch(
        "intergrax.autonomous_work.postgresql_repository.PostgreSQLConnectionProvider",
        return_value=provider,
    ) as provider_ctor:
        store = PostgreSQLAutonomousWorkStore(
            config,
            connection_factory=lambda: conn,
            schema_name=schema_name,
        )
    provider_ctor.assert_called_once()
    return store, provider


def test_autonomous_work_store_uses_platform_connection_provider() -> None:
    conn = _FakeConnection()
    config = PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test", tenant_schema="tenant_a")

    store, provider = _open_store_with_injected_provider(
        config,
        conn,
        schema_name="tenant_a",
    )

    assert provider.tenant_schema == "tenant_a"
    assert conn.committed >= 1
    assert conn.executed
    store.close()


def test_autonomous_work_adapter_does_not_import_psycopg() -> None:
    imported = _import_names(_AW_POSTGRESQL_ADAPTER)
    assert "psycopg" not in imported


def test_opens_and_autonomous_work_share_platform_provider() -> None:
    conn = _FakeConnection()
    config = PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test", tenant_schema="tenant_a")

    with patch.object(PostgreSQLConnectionProvider, "open_configured_connection", return_value=conn):
        open_postgresql_relational_store(config, connection_factory=lambda: conn)

    store, provider = _open_store_with_injected_provider(
        config,
        conn,
        schema_name="tenant_a",
    )
    store.close()

    assert provider.tenant_schema == "tenant_a"


def test_invalid_schema_rejected_before_autonomous_work_store_open() -> None:
    config = PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test")
    with pytest.raises(ValueError, match="tenant_schema must be a simple SQL identifier"):
        PostgreSQLAutonomousWorkStore(config, schema_name="bad-schema")
