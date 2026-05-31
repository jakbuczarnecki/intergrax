# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Databricks connection openers — internal to the databricks integration package.

Only this module may call ``databricks.sql.connect``. All composition roots use
``bundle.create_databricks_*`` factories or ``profile.resolve(RELATIONAL_STORE)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.providers.relational_store.databricks.adapter import DatabricksRelationalStore
from intergrax.integrations.providers.relational_store.databricks.config import DatabricksIntegrationConfig


def _import_databricks_sql() -> Any:
    try:
        from databricks import sql
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "Databricks integration requires databricks-sql-connector. "
            "Install with: uv sync --extra dev  (includes databricks-sql-connector)"
        ) from exc
    return sql


def _open_connection(
    config: DatabricksIntegrationConfig,
    *,
    connection_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    if connection_factory is not None:
        return connection_factory()
    sql = _import_databricks_sql()
    try:
        return sql.connect(**config.connect_kwargs())
    except ValueError as exc:
        raise IntegrationConfigurationError(str(exc)) from exc


def _apply_unity_catalog(
    connection: Any,
    *,
    catalog: str | None,
    tenant_schema: str | None,
) -> None:
    if catalog:
        with connection.cursor() as cursor:
            cursor.execute(f"USE CATALOG {catalog}")
    if tenant_schema:
        with connection.cursor() as cursor:
            cursor.execute(f"USE SCHEMA {tenant_schema}")


def open_databricks_relational_store(
    config: DatabricksIntegrationConfig,
    *,
    implementation: Optional[RelationalStore] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
) -> RelationalStore:
    if implementation is not None:
        return implementation
    connection = _open_connection(config, connection_factory=connection_factory)
    if config.catalog or config.tenant_schema:
        _apply_unity_catalog(
            connection,
            catalog=config.catalog,
            tenant_schema=config.tenant_schema,
        )
    return DatabricksRelationalStore(config, connection)
