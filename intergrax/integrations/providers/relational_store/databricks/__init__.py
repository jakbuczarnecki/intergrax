# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Databricks integration — SQL Warehouse ``RelationalStore`` (Phase M.6 P2).

Public entry points: ``create_databricks_relational_store()``, ``create_databricks_integration()``,
``register_databricks_integration()``, and ``profile.resolve(RELATIONAL_STORE)``.
"""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.relational_store.databricks.config import (
    ENV_DATABRICKS_CATALOG,
    ENV_DATABRICKS_HOST,
    ENV_DATABRICKS_HTTP_PATH,
    ENV_DATABRICKS_SCHEMA,
    ENV_DATABRICKS_TOKEN,
    DatabricksIntegrationConfig,
)

__all__ = [
    "ENV_DATABRICKS_CATALOG",
    "ENV_DATABRICKS_HOST",
    "ENV_DATABRICKS_HTTP_PATH",
    "ENV_DATABRICKS_SCHEMA",
    "ENV_DATABRICKS_TOKEN",
    "DatabricksIntegrationBundle",
    "DatabricksIntegrationConfig",
    "DatabricksRelationalStore",
    "create_databricks_integration",
    "create_databricks_relational_store",
    "register_databricks_integration",
    "resolve_databricks_config",
    "create_databricks_relational_store_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "DatabricksIntegrationBundle",
        "DatabricksRelationalStore",
        "create_databricks_integration",
        "create_databricks_relational_store",
        "register_databricks_integration",
        "resolve_databricks_config",
        "create_databricks_relational_store_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "DATABRICKS_RELATIONAL_STORE_PROVIDER_ID",
        "DatabricksRelationalStoreIntegration",
        "DatabricksRelationalStoreIntegrationConfig",
        "DatabricksRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_databricks_integration":
        from intergrax.integrations.providers.relational_store.databricks.register import register_databricks_integration

        return register_databricks_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.relational_store.databricks import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "DatabricksRelationalStore":
        from intergrax.integrations.providers.relational_store.databricks.adapter import _DatabricksRelationalStore

        return DatabricksRelationalStore
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.databricks import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
