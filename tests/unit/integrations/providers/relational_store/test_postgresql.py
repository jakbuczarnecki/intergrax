# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for PostgreSQL integration provider (Phase M.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_relational_store
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.postgresql.adapter import PostgreSQLRelationalStore
from intergrax.integrations.providers.relational_store.postgresql.bundle import (
    PostgreSQLIntegrationBundle,
    create_postgresql_integration,
    create_postgresql_relational_store,
)
from intergrax.integrations.providers.relational_store.postgresql.config import (
    ENV_POSTGRESQL_DSN,
    ENV_POSTGRESQL_HOST,
    ENV_POSTGRESQL_TENANT_SCHEMA,
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.register import register_postgresql_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_POSTGRESQL_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "postgresql"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "import psycopg",
    "from psycopg",
    "PostgreSQLRelationalStore(",
    "integrations.providers.postgresql.adapter",
    "integrations.providers.postgresql.opens",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[dict[str, Any]]:
        return self._rows


class _FakeConnection:
    def __init__(self, *, fetch_rows: list[dict[str, Any]] | None = None) -> None:
        self.executed: list[tuple[str, Sequence[Any]]] = []
        self._fetch_rows = fetch_rows or []
        self.committed = 0
        self.closed = False

    def execute(self, sql: str, params: Sequence[Any] = ()) -> _FakeCursor:
        self.executed.append((sql, params))
        return _FakeCursor(self._fetch_rows)

    def commit(self) -> None:
        self.committed += 1

    def close(self) -> None:
        self.closed = True


def _connection_factory(rows: list[dict[str, Any]] | None = None):
    conn = _FakeConnection(fetch_rows=rows or [{"name": "alpha"}])

    def _factory() -> _FakeConnection:
        return conn

    return _factory, conn


def _iter_python_files(*roots: str):
    for root_name in roots:
        root = _PROJECT_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            yield path


def test_psycopg_connect_only_in_opens_module() -> None:
    violations: list[str] = []
    for path in _POSTGRESQL_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "psycopg" in text:
            violations.append(path.name)
    assert violations == []


def test_postgresql_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _POSTGRESQL_PKG in path.parents:
            continue

        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_postgresql_config_from_env_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_POSTGRESQL_DSN, "postgresql://u:p@db.example:5432/app")
    config = PostgreSQLIntegrationConfig.from_env()
    assert config.connection_string() == "postgresql://u:p@db.example:5432/app"


def test_postgresql_config_builds_connection_string_from_parts() -> None:
    config = PostgreSQLIntegrationConfig(
        user="app",
        password="s3cret!",
        host="postgres.internal",
        port=5433,
        database="intergrax",
        sslmode="require",
    )
    assert config.connection_string() == (
        "postgresql://app:s3cret%21@postgres.internal:5433/intergrax?sslmode=require"
    )


def test_postgresql_config_rejects_invalid_tenant_schema() -> None:
    with pytest.raises(ValueError, match="tenant_schema must be a simple SQL identifier"):
        PostgreSQLIntegrationConfig(tenant_schema="bad-schema")


def test_postgresql_config_from_env_host_and_tenant_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_POSTGRESQL_HOST, "127.0.0.1")
    monkeypatch.setenv(ENV_POSTGRESQL_TENANT_SCHEMA, "tenant_a")
    config = PostgreSQLIntegrationConfig.from_env()
    assert config.host == "127.0.0.1"
    assert config.tenant_schema == "tenant_a"


def test_postgresql_relational_store_execute_and_fetch() -> None:
    factory, conn = _connection_factory()
    store = create_postgresql_relational_store(
        connection_factory=factory,
        dsn="postgresql://localhost/test",
    )
    assert_relational_store(store)

    store.execute("CREATE TABLE items (id serial primary key, name text not null)")
    store.execute("INSERT INTO items (name) VALUES (%s)", ("alpha",))
    rows = store.fetch_all("SELECT name FROM items")
    store.close()

    assert [row["name"] for row in rows] == ["alpha"]
    assert conn.executed[0][0].startswith("CREATE TABLE items")
    assert conn.executed[1] == ("INSERT INTO items (name) VALUES (%s)", ("alpha",))
    assert conn.closed is True


def test_postgresql_opens_sets_search_path_when_tenant_schema_configured() -> None:
    factory, conn = _connection_factory()
    store = create_postgresql_relational_store(
        connection_factory=factory,
        dsn="postgresql://localhost/test",
        tenant_schema="tenant_a",
    )
    assert isinstance(store, PostgreSQLRelationalStore)

    assert conn.executed == [("SET search_path TO tenant_a, public", ())]
    assert conn.committed == 1


def test_postgresql_opens_uses_psycopg_when_no_connection_factory() -> None:
    config = PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test", tenant_schema="tenant_a")
    mock_conn = MagicMock()
    mock_psycopg = MagicMock()
    mock_psycopg.connect.return_value = mock_conn
    dict_row = MagicMock()

    with patch(
        "intergrax.integrations.providers.relational_store.postgresql.opens._import_psycopg",
        return_value=(mock_psycopg, dict_row),
    ):
        from intergrax.integrations.providers.relational_store.postgresql.opens import open_postgresql_relational_store

        store = open_postgresql_relational_store(config)

    assert isinstance(store, PostgreSQLRelationalStore)
    mock_psycopg.connect.assert_called_once_with(
        "postgresql://localhost/test",
        row_factory=dict_row,
    )
    mock_conn.execute.assert_called_once_with("SET search_path TO tenant_a, public")
    assert mock_conn.commit.call_count == 1


def test_postgresql_missing_psycopg_raises_configuration_error() -> None:
    config = PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test")

    with patch(
        "intergrax.integrations.providers.relational_store.postgresql.opens._import_psycopg",
        side_effect=IntegrationConfigurationError("missing psycopg"),
    ):
        from intergrax.integrations.providers.relational_store.postgresql.opens import open_postgresql_relational_store

        with pytest.raises(IntegrationConfigurationError, match="missing psycopg"):
            open_postgresql_relational_store(config)


def test_create_postgresql_integration_bundle() -> None:
    factory, _conn = _connection_factory()
    bundle = create_postgresql_integration(
        connection_factory=factory,
        dsn="postgresql://localhost/test",
    )

    assert isinstance(bundle, PostgreSQLIntegrationBundle)
    assert isinstance(bundle.relational_store, PostgreSQLRelationalStore)
    assert bundle.config.dsn == "postgresql://localhost/test"


def test_register_and_resolve_via_profile() -> None:
    register_postgresql_integration()
    profile = IntegrationProfile(relational_store=IntegrationSlug.POSTGRESQL)
    factory, _conn = _connection_factory()

    store = resolve(
        IntegrationCategory.RELATIONAL_STORE,
        profile=profile,
        config={"dsn": "postgresql://localhost/test", "connection_factory": factory},
    )

    assert_relational_store(store)
    assert isinstance(store, PostgreSQLRelationalStore)


def test_register_default_integrations_includes_postgresql() -> None:
    register_default_integrations()
    profile = IntegrationProfile(relational_store=IntegrationSlug.POSTGRESQL)
    factory, _conn = _connection_factory()

    store = resolve(
        IntegrationCategory.RELATIONAL_STORE,
        profile=profile,
        config={"dsn": "postgresql://localhost/test", "connection_factory": factory},
    )

    assert isinstance(store, PostgreSQLRelationalStore)


def test_create_postgresql_relational_store_catalog_factory() -> None:
    factory, _conn = _connection_factory()
    store = create_postgresql_relational_store(
        connection_factory=factory,
        dsn="postgresql://localhost/test",
    )
    assert_relational_store(store)
    assert store.config.dsn == "postgresql://localhost/test"
