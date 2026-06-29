# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level PostgreSQL connection openers — internal to the postgresql integration package.

Only this module may call ``psycopg.connect``. All composition roots use
``bundle.create_postgresql_*`` factories or ``profile.resolve(RELATIONAL_STORE)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.providers.relational_store.postgresql.adapter import _PostgreSQLRelationalStore
from intergrax.integrations.providers.relational_store.postgresql.integration import PostgresqlRelationalStoreIntegration
from intergrax.integrations.providers.relational_store.postgresql.config import PostgreSQLIntegrationConfig


def _import_psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "PostgreSQL integration requires psycopg. "
            "Install with: uv sync --extra dev  (includes psycopg[binary])"
        ) from exc
    return psycopg, dict_row


def _open_connection(
    config: PostgreSQLIntegrationConfig,
    *,
    connection_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    if connection_factory is not None:
        return connection_factory()
    psycopg, dict_row = _import_psycopg()
    return psycopg.connect(
        config.connection_string(),
        row_factory=dict_row,
    )


def _apply_search_path(connection: Any, tenant_schema: str) -> None:
    connection.execute(f"SET search_path TO {tenant_schema}, public")
    connection.commit()


def open_postgresql_relational_store(
    config: PostgreSQLIntegrationConfig,
    *,
    implementation: Optional[RelationalStore] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
) -> RelationalStore:
    if implementation is not None:
        return implementation
    connection = _open_connection(config, connection_factory=connection_factory)
    if config.tenant_schema:
        _apply_search_path(connection, config.tenant_schema)
    return PostgresqlRelationalStoreIntegration.from_runtime(_PostgreSQLRelationalStore(config, connection))