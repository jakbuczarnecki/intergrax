# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Databricks integration provider (Phase M.6 P2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

import pytest

from intergrax.integrations._shared.conformance import assert_relational_store
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.databricks.adapter import DatabricksRelationalStore
from intergrax.integrations.providers.relational_store.databricks.bundle import (
    DatabricksIntegrationBundle,
    create_databricks_integration,
    create_databricks_relational_store,
)
from intergrax.integrations.providers.relational_store.databricks.config import (
    ENV_DATABRICKS_HOST,
    ENV_DATABRICKS_HTTP_PATH,
    ENV_DATABRICKS_SCHEMA,
    ENV_DATABRICKS_TOKEN,
    DatabricksIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.databricks.register import register_databricks_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATABRICKS_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "databricks"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "from databricks import sql",
    "import databricks",
    "DatabricksRelationalStore(",
    "integrations.providers.databricks.adapter",
    "integrations.providers.databricks.opens",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeCursor:
    def __init__(self, *, fetch_rows: list[tuple[Any, ...]] | None = None) -> None:
        self.executed: list[tuple[str, Sequence[Any] | None]] = []
        self._fetch_rows = fetch_rows or [("alpha",)]
        self.description = [("name",)]

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._fetch_rows


class _FakeConnection:
    def __init__(self, *, fetch_rows: list[tuple[Any, ...]] | None = None) -> None:
        self._cursor = _FakeCursor(fetch_rows=fetch_rows)
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def close(self) -> None:
        self.closed = True


def _connection_factory(rows: list[tuple[Any, ...]] | None = None):
    conn = _FakeConnection(fetch_rows=rows or [("alpha",)])

    def _factory() -> _FakeConnection:
        return conn

    return _factory, conn


def _databricks_config(**overrides: object) -> DatabricksIntegrationConfig:
    base = {
        "host": "adb-123.4.azuredatabricks.net",
        "http_path": "/sql/1.0/warehouses/abc",
        "access_token": "dapi-test",
    }
    base.update(overrides)
    return DatabricksIntegrationConfig.model_validate(base)


def _iter_python_files(*roots: str):
    for root_name in roots:
        root = _PROJECT_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            yield path


def test_databricks_sql_connect_only_in_opens_module() -> None:
    violations: list[str] = []
    for path in _DATABRICKS_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "databricks" in text:
            violations.append(path.name)
    assert violations == []


def test_databricks_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _DATABRICKS_PKG in path.parents:
            continue

        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_databricks_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_DATABRICKS_HOST, "https://adb-123.4.azuredatabricks.net/")
    monkeypatch.setenv(ENV_DATABRICKS_HTTP_PATH, "/sql/1.0/warehouses/abc")
    monkeypatch.setenv(ENV_DATABRICKS_TOKEN, "dapi-test")
    monkeypatch.setenv(ENV_DATABRICKS_SCHEMA, "analytics")
    config = DatabricksIntegrationConfig.from_env()
    assert config.host == "adb-123.4.azuredatabricks.net"
    assert config.http_path == "/sql/1.0/warehouses/abc"
    assert config.access_token == "dapi-test"
    assert config.tenant_schema == "analytics"


def test_databricks_config_connect_kwargs() -> None:
    config = _databricks_config()
    assert config.connect_kwargs() == {
        "server_hostname": "adb-123.4.azuredatabricks.net",
        "http_path": "/sql/1.0/warehouses/abc",
        "access_token": "dapi-test",
    }


def test_databricks_config_rejects_invalid_tenant_schema() -> None:
    with pytest.raises(ValidationError, match="simple SQL identifiers"):
        DatabricksIntegrationConfig(
            host="adb.example.net",
            http_path="/sql/1.0/warehouses/x",
            access_token="token",
            tenant_schema="bad-schema",
        )


def test_databricks_relational_store_execute_and_fetch() -> None:
    factory, conn = _connection_factory()
    store = create_databricks_relational_store(
        connection_factory=factory,
        host="adb.example.net",
        http_path="/sql/1.0/warehouses/x",
        access_token="token",
    )
    assert_relational_store(store)

    store.execute("CREATE TABLE items (id INT, name STRING)")
    store.execute("INSERT INTO items VALUES (?, ?)", (1, "alpha"))
    rows = store.fetch_all("SELECT name FROM items")
    store.close()

    assert [row["name"] for row in rows] == ["alpha"]
    assert conn._cursor.executed[0] == ("CREATE TABLE items (id INT, name STRING)", None)
    assert conn._cursor.executed[1] == ("INSERT INTO items VALUES (?, ?)", (1, "alpha"))
    assert conn.closed is True


def test_databricks_opens_sets_unity_catalog_when_configured() -> None:
    factory, conn = _connection_factory()
    store = create_databricks_relational_store(
        connection_factory=factory,
        host="adb.example.net",
        http_path="/sql/1.0/warehouses/x",
        access_token="token",
        catalog="main",
        tenant_schema="analytics",
    )
    assert isinstance(store, DatabricksRelationalStore)

    assert conn._cursor.executed == [
        ("USE CATALOG main", None),
        ("USE SCHEMA analytics", None),
    ]


def test_databricks_opens_uses_driver_when_no_connection_factory() -> None:
    config = _databricks_config(catalog="main", tenant_schema="analytics")
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_sql = MagicMock()
    mock_sql.connect.return_value = mock_conn

    with patch(
        "intergrax.integrations.providers.relational_store.databricks.opens._import_databricks_sql",
        return_value=mock_sql,
    ):
        from intergrax.integrations.providers.relational_store.databricks.opens import open_databricks_relational_store

        store = open_databricks_relational_store(config)

    assert isinstance(store, DatabricksRelationalStore)
    mock_sql.connect.assert_called_once_with(
        server_hostname="adb-123.4.azuredatabricks.net",
        http_path="/sql/1.0/warehouses/abc",
        access_token="dapi-test",
    )
    assert mock_cursor.execute.call_args_list[0].args == ("USE CATALOG main",)
    assert mock_cursor.execute.call_args_list[1].args == ("USE SCHEMA analytics",)


def test_databricks_missing_driver_raises_configuration_error() -> None:
    config = _databricks_config()

    with patch(
        "intergrax.integrations.providers.relational_store.databricks.opens._import_databricks_sql",
        side_effect=IntegrationConfigurationError("missing databricks-sql-connector"),
    ):
        from intergrax.integrations.providers.relational_store.databricks.opens import open_databricks_relational_store

        with pytest.raises(IntegrationConfigurationError, match="missing databricks-sql-connector"):
            open_databricks_relational_store(config)


def test_create_databricks_integration_bundle() -> None:
    factory, _conn = _connection_factory()
    bundle = create_databricks_integration(
        connection_factory=factory,
        host="adb.example.net",
        http_path="/sql/1.0/warehouses/x",
        access_token="token",
    )

    assert isinstance(bundle, DatabricksIntegrationBundle)
    assert isinstance(bundle.relational_store, DatabricksRelationalStore)
    assert bundle.config.host == "adb.example.net"


def test_register_and_resolve_via_profile() -> None:
    register_databricks_integration()
    profile = IntegrationProfile(relational_store=IntegrationSlug.DATABRICKS)
    factory, _conn = _connection_factory()

    store = resolve(
        IntegrationCategory.RELATIONAL_STORE,
        profile=profile,
        config={
            "host": "adb.example.net",
            "http_path": "/sql/1.0/warehouses/x",
            "access_token": "token",
            "connection_factory": factory,
        },
    )

    assert_relational_store(store)
    assert isinstance(store, DatabricksRelationalStore)


def test_register_default_integrations_includes_databricks() -> None:
    register_default_integrations()
    profile = IntegrationProfile(relational_store=IntegrationSlug.DATABRICKS)
    factory, _conn = _connection_factory()

    store = resolve(
        IntegrationCategory.RELATIONAL_STORE,
        profile=profile,
        config={
            "host": "adb.example.net",
            "http_path": "/sql/1.0/warehouses/x",
            "access_token": "token",
            "connection_factory": factory,
        },
    )

    assert isinstance(store, DatabricksRelationalStore)


def test_create_databricks_relational_store_catalog_factory() -> None:
    factory, _conn = _connection_factory()
    store = create_databricks_relational_store(
        connection_factory=factory,
        host="adb.example.net",
        http_path="/sql/1.0/warehouses/x",
        access_token="token",
    )
    assert_relational_store(store)
    assert store.config.access_token == "token"
