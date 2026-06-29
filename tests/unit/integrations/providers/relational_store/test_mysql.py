# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for MySQL integration provider (Phase M.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_relational_store
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.mysql.integration import MysqlRelationalStoreIntegration
from intergrax.integrations.providers.relational_store.mysql.bundle import (
    MySQLIntegrationBundle,
    create_mysql_integration,
    create_mysql_relational_store,
)
from intergrax.integrations.providers.relational_store.mysql.config import (
    ENV_MYSQL_DSN,
    ENV_MYSQL_HOST,
    ENV_MYSQL_TENANT_DATABASE,
    MySQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.mysql.register import register_mysql_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MYSQL_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "mysql"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "import pymysql",
    "from pymysql",
    "MysqlRelationalStoreIntegration(",
    "integrations.providers.mysql.adapter",
    "integrations.providers.mysql.opens",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeCursor:
    def __init__(self, *, fetch_rows: list[dict[str, Any]] | None = None) -> None:
        self.executed: list[tuple[str, Sequence[Any]]] = []
        self._fetch_rows = fetch_rows or []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> list[dict[str, Any]]:
        return self._fetch_rows


class _FakeConnection:
    def __init__(self, *, fetch_rows: list[dict[str, Any]] | None = None) -> None:
        self._cursor = _FakeCursor(fetch_rows=fetch_rows or [{"name": "alpha"}])
        self.committed = 0
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return self._cursor

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


def test_pymysql_connect_only_in_opens_module() -> None:
    violations: list[str] = []
    for path in _MYSQL_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "pymysql" in text:
            violations.append(path.name)
    assert violations == []


def test_mysql_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _MYSQL_PKG in path.parents:
            continue

        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_mysql_config_from_env_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MYSQL_DSN, "mysql://u:p@db.example:3306/app")
    config = MySQLIntegrationConfig.from_env()
    assert config.connection_string() == "mysql://u:p@db.example:3306/app"


def test_mysql_config_builds_connection_string_from_parts() -> None:
    config = MySQLIntegrationConfig(
        user="app",
        password="s3cret!",
        host="mysql.internal",
        port=3307,
        database="intergrax",
        charset="utf8mb4",
    )
    assert config.connection_string() == (
        "mysql://app:s3cret%21@mysql.internal:3307/intergrax?charset=utf8mb4"
    )


def test_mysql_config_connection_kwargs_from_dsn() -> None:
    config = MySQLIntegrationConfig(dsn="mysql://app:pass@db:3306/demo?charset=utf8")
    assert config.connection_kwargs() == {
        "host": "db",
        "port": 3306,
        "user": "app",
        "password": "pass",
        "database": "demo",
        "charset": "utf8",
    }


def test_mysql_config_rejects_invalid_tenant_database() -> None:
    with pytest.raises(ValueError, match="tenant_database must be a simple SQL identifier"):
        MySQLIntegrationConfig(tenant_database="bad-db")


def test_mysql_config_from_env_host_and_tenant_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MYSQL_HOST, "127.0.0.1")
    monkeypatch.setenv(ENV_MYSQL_TENANT_DATABASE, "tenant_a")
    config = MySQLIntegrationConfig.from_env()
    assert config.host == "127.0.0.1"
    assert config.tenant_database == "tenant_a"


def test_mysql_relational_store_execute_and_fetch() -> None:
    factory, conn = _connection_factory()
    store = create_mysql_relational_store(
        connection_factory=factory,
        dsn="mysql://localhost/test",
    )
    assert_relational_store(store)

    store.execute("CREATE TABLE items (id INT PRIMARY KEY, name VARCHAR(64) NOT NULL)")
    store.execute("INSERT INTO items (name) VALUES (%s)", ("alpha",))
    rows = store.fetch_all("SELECT name FROM items")
    store.close()

    assert [row["name"] for row in rows] == ["alpha"]
    assert conn._cursor.executed[0][0].startswith("CREATE TABLE items")
    assert conn._cursor.executed[1] == ("INSERT INTO items (name) VALUES (%s)", ("alpha",))
    assert conn.closed is True


def test_mysql_opens_selects_tenant_database() -> None:
    factory, conn = _connection_factory()
    store = create_mysql_relational_store(
        connection_factory=factory,
        dsn="mysql://localhost/test",
        tenant_database="tenant_a",
    )
    assert isinstance(store, MysqlRelationalStoreIntegration)
    assert conn._cursor.executed == [("USE `tenant_a`", ())]
    assert conn.committed == 1


def test_mysql_opens_uses_pymysql_when_no_connection_factory() -> None:
    config = MySQLIntegrationConfig(dsn="mysql://localhost/test", tenant_database="tenant_a")
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_pymysql = MagicMock()
    mock_pymysql.connect.return_value = mock_conn
    dict_cursor = MagicMock()

    with patch(
        "intergrax.integrations.providers.relational_store.mysql.opens._import_pymysql",
        return_value=(mock_pymysql, dict_cursor),
    ):
        from intergrax.integrations.providers.relational_store.mysql.opens import open_mysql_relational_store

        store = open_mysql_relational_store(config)

    assert isinstance(store, MysqlRelationalStoreIntegration)
    mock_pymysql.connect.assert_called_once()
    connect_kwargs = mock_pymysql.connect.call_args.kwargs
    assert connect_kwargs["cursorclass"] is dict_cursor
    assert connect_kwargs["autocommit"] is False
    mock_cursor.execute.assert_called_once_with("USE `tenant_a`")


def test_mysql_missing_pymysql_raises_configuration_error() -> None:
    config = MySQLIntegrationConfig(dsn="mysql://localhost/test")

    with patch(
        "intergrax.integrations.providers.relational_store.mysql.opens._import_pymysql",
        side_effect=IntegrationConfigurationError("missing pymysql"),
    ):
        from intergrax.integrations.providers.relational_store.mysql.opens import open_mysql_relational_store

        with pytest.raises(IntegrationConfigurationError, match="missing pymysql"):
            open_mysql_relational_store(config)


def test_create_mysql_integration_bundle() -> None:
    factory, _conn = _connection_factory()
    bundle = create_mysql_integration(
        connection_factory=factory,
        dsn="mysql://localhost/test",
    )

    assert isinstance(bundle, MySQLIntegrationBundle)
    assert isinstance(bundle.relational_store, MysqlRelationalStoreIntegration)
    assert bundle.config.dsn == "mysql://localhost/test"


def test_register_and_resolve_via_profile() -> None:
    register_mysql_integration()
    profile = IntegrationProfile(relational_store="mysql")
    factory, _conn = _connection_factory()

    store = resolve(
        IntegrationCategory.RELATIONAL_STORE,
        profile=profile,
        config={"dsn": "mysql://localhost/test", "connection_factory": factory},
    )

    assert_relational_store(store)
    assert isinstance(store, MysqlRelationalStoreIntegration)


def test_register_default_integrations_includes_mysql() -> None:
    register_default_integrations()
    profile = IntegrationProfile(relational_store="mysql")
    factory, _conn = _connection_factory()

    store = resolve(
        IntegrationCategory.RELATIONAL_STORE,
        profile=profile,
        config={"dsn": "mysql://localhost/test", "connection_factory": factory},
    )

    assert isinstance(store, MysqlRelationalStoreIntegration)


def test_create_mysql_relational_store_catalog_factory() -> None:
    factory, _conn = _connection_factory()
    store = create_mysql_relational_store(
        connection_factory=factory,
        dsn="mysql://localhost/test",
    )
    assert_relational_store(store)
    assert store.config.dsn == "mysql://localhost/test"
