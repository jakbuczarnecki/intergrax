# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level PostgreSQL connection openers — internal to the postgresql integration package.

Composition roots use ``bundle.create_postgresql_*`` factories or ``profile.resolve(RELATIONAL_STORE)``.
Driver I/O is owned by ``session.PostgreSQLConnectionProvider``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.providers.relational_store.postgresql.adapter import _PostgreSQLRelationalStore
from intergrax.integrations.providers.relational_store.postgresql.integration import PostgresqlRelationalStoreIntegration
from intergrax.integrations.providers.relational_store.postgresql.config import PostgreSQLIntegrationConfig
from intergrax.integrations.providers.relational_store.postgresql.session import PostgreSQLConnectionProvider


def open_postgresql_relational_store(
    config: PostgreSQLIntegrationConfig,
    *,
    implementation: Optional[RelationalStore] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
) -> PostgresqlRelationalStoreIntegration:
    if implementation is not None:
        if isinstance(implementation, PostgresqlRelationalStoreIntegration):
            return implementation
        return PostgresqlRelationalStoreIntegration.from_client(implementation)
    provider = PostgreSQLConnectionProvider(
        config,
        connection_factory=connection_factory,
        tenant_schema=config.tenant_schema,
    )
    connection = provider.open_configured_connection()
    return PostgresqlRelationalStoreIntegration.from_client(_PostgreSQLRelationalStore(config, connection))
