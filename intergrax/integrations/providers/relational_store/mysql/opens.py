# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level MySQL connection openers — internal to the mysql integration package.

Only this module may call ``pymysql.connect``. All composition roots use
``bundle.create_mysql_*`` factories or ``profile.resolve(RELATIONAL_STORE)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.providers.relational_store.mysql.adapter import MySQLRelationalStore
from intergrax.integrations.providers.relational_store.mysql.config import MySQLIntegrationConfig


def _import_pymysql() -> tuple[Any, Any]:
    try:
        import pymysql
        from pymysql.cursors import DictCursor
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "MySQL integration requires pymysql. "
            "Install with: uv sync --extra dev  (includes pymysql)"
        ) from exc
    return pymysql, DictCursor


def _open_connection(
    config: MySQLIntegrationConfig,
    *,
    connection_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    if connection_factory is not None:
        return connection_factory()
    pymysql, dict_cursor = _import_pymysql()
    kwargs = config.connection_kwargs()
    return pymysql.connect(
        **kwargs,
        cursorclass=dict_cursor,
        autocommit=False,
    )


def _select_tenant_database(connection: Any, tenant_database: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute(f"USE `{tenant_database}`")
    connection.commit()


def open_mysql_relational_store(
    config: MySQLIntegrationConfig,
    *,
    implementation: Optional[RelationalStore] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
) -> RelationalStore:
    if implementation is not None:
        return implementation
    connection = _open_connection(config, connection_factory=connection_factory)
    if config.tenant_database:
        _select_tenant_database(connection, config.tenant_database)
    return MySQLRelationalStore(config, connection)
