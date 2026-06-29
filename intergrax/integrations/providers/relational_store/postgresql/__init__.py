# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
PostgreSQL integration — production ``RelationalStore`` (Phase M.6).

Public entry points: ``create_postgresql_relational_store()``, ``create_postgresql_integration()``,
``register_postgresql_integration()``, and ``profile.resolve(RELATIONAL_STORE)``.
"""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.relational_store.postgresql.config import (
    ENV_POSTGRESQL_DATABASE,
    ENV_POSTGRESQL_DSN,
    ENV_POSTGRESQL_HOST,
    ENV_POSTGRESQL_PASSWORD,
    ENV_POSTGRESQL_PORT,
    ENV_POSTGRESQL_TENANT_SCHEMA,
    ENV_POSTGRESQL_SSLMODE,
    ENV_POSTGRESQL_USER,
    PostgreSQLIntegrationConfig,
)

__all__ = [
    "ENV_POSTGRESQL_DATABASE",
    "ENV_POSTGRESQL_DSN",
    "ENV_POSTGRESQL_HOST",
    "ENV_POSTGRESQL_PASSWORD",
    "ENV_POSTGRESQL_PORT",
    "ENV_POSTGRESQL_TENANT_SCHEMA",
    "ENV_POSTGRESQL_SSLMODE",
    "ENV_POSTGRESQL_USER",
    "PostgreSQLIntegrationBundle",
    "PostgreSQLIntegrationConfig",
    "PostgreSQLRelationalStore",
    "create_postgresql_integration",
    "create_postgresql_relational_store",
    "register_postgresql_integration",
    "resolve_postgresql_config",
    "create_postgresql_relational_store_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "PostgreSQLIntegrationBundle",
        "PostgreSQLRelationalStore",
        "create_postgresql_integration",
        "create_postgresql_relational_store",
        "register_postgresql_integration",
        "resolve_postgresql_config",
        "create_postgresql_relational_store_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "POSTGRESQL_RELATIONAL_STORE_PROVIDER_ID",
        "PostgresqlRelationalStoreIntegration",
        "PostgresqlRelationalStoreIntegrationConfig",
        "PostgresqlRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_postgresql_integration":
        from intergrax.integrations.providers.relational_store.postgresql.register import register_postgresql_integration

        return register_postgresql_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.relational_store.postgresql import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "PostgreSQLRelationalStore":
        from intergrax.integrations.providers.relational_store.postgresql.adapter import _PostgreSQLRelationalStore

        return PostgreSQLRelationalStore
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.postgresql import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
