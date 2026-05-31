# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
PostgreSQL integration — production ``RelationalStore`` (Phase M.6).

Public entry points: ``create_postgresql_relational_store()``, ``create_postgresql_integration()``,
``register_postgresql_integration()``, and ``profile.resolve(RELATIONAL_STORE)``.
"""

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
]

_LAZY_EXPORTS = frozenset(
    {
        "PostgreSQLIntegrationBundle",
        "PostgreSQLRelationalStore",
        "create_postgresql_integration",
        "create_postgresql_relational_store",
        "register_postgresql_integration",
        "resolve_postgresql_config",
    }
)


def __getattr__(name: str):
    if name == "register_postgresql_integration":
        from intergrax.integrations.providers.relational_store.postgresql.register import register_postgresql_integration

        return register_postgresql_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.relational_store.postgresql import bundle as _bundle

        return getattr(_bundle, name)
    if name == "PostgreSQLRelationalStore":
        from intergrax.integrations.providers.relational_store.postgresql.adapter import PostgreSQLRelationalStore

        return PostgreSQLRelationalStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
