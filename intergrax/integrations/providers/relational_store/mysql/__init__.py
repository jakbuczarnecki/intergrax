# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
MySQL integration — production ``RelationalStore`` (Phase M.6).

Public entry points: ``create_mysql_relational_store()``, ``create_mysql_integration()``,
``register_mysql_integration()``, and ``profile.resolve(RELATIONAL_STORE)``.
"""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.relational_store.mysql.config import (
    ENV_MYSQL_CHARSET,
    ENV_MYSQL_DATABASE,
    ENV_MYSQL_DSN,
    ENV_MYSQL_HOST,
    ENV_MYSQL_PASSWORD,
    ENV_MYSQL_PORT,
    ENV_MYSQL_TENANT_DATABASE,
    ENV_MYSQL_USER,
    MySQLIntegrationConfig,
)

__all__ = [
    "ENV_MYSQL_CHARSET",
    "ENV_MYSQL_DATABASE",
    "ENV_MYSQL_DSN",
    "ENV_MYSQL_HOST",
    "ENV_MYSQL_PASSWORD",
    "ENV_MYSQL_PORT",
    "ENV_MYSQL_TENANT_DATABASE",
    "ENV_MYSQL_USER",
    "MySQLIntegrationBundle",
    "MySQLIntegrationConfig",
    "MySQLRelationalStore",
    "create_mysql_integration",
    "create_mysql_relational_store",
    "register_mysql_integration",
    "resolve_mysql_config",
    "create_mysql_relational_store_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "MySQLIntegrationBundle",
        "MySQLRelationalStore",
        "create_mysql_integration",
        "create_mysql_relational_store",
        "register_mysql_integration",
        "resolve_mysql_config",
        "create_mysql_relational_store_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "MYSQL_RELATIONAL_STORE_PROVIDER_ID",
        "MysqlRelationalStoreIntegration",
        "MysqlRelationalStoreIntegrationConfig",
        "MysqlRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_mysql_integration":
        from intergrax.integrations.providers.relational_store.mysql.register import register_mysql_integration

        return register_mysql_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.relational_store.mysql import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "MySQLRelationalStore":
        from intergrax.integrations.providers.relational_store.mysql.adapter import _MySQLRelationalStore

        return MySQLRelationalStore
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.mysql import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
