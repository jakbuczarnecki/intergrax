# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
SQLite integration — single public entry for all SQLite-backed Tier-0 facades.

Domain store classes live under ``intergrax.runtime.*`` and ``intergrax.experiments``;
compose them only through this package.
"""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.relational_store.sqlite.adapter import _SQLiteRelationalStore
from intergrax.integrations.providers.relational_store.sqlite.config import (
    ENV_SQLITE_DATA_DIR,
    SQLiteIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    DEFAULT_EXPERIMENTS_DB,
    DEFAULT_HUMAN_DECISIONS_DB,
    DEFAULT_RUNTIME_EVENTS_DB,
    DEFAULT_TASK_CHECKPOINTS_DB,
    DEFAULT_TASK_MEMORY_DB,
    DEFAULT_TRACE_DB,
    ENV_EXPERIMENTS_DB,
    ENV_HUMAN_DECISIONS_DB,
    ENV_IDEMPOTENCY_DB,
    ENV_ORGANIZATION_DB,
    ENV_RELATIONAL_DB,
    ENV_RUNTIME_EVENTS_DB,
    ENV_SESSION_DB,
    ENV_TASK_CHECKPOINTS_DB,
    ENV_TASK_MEMORY_DB,
    ENV_TRACE_DB,
    ENV_USER_PROFILE_DB,
    SqliteStorePaths,
    resolve_experiments_db_path,
    resolve_human_decisions_db_path,
    resolve_idempotency_db_path,
    resolve_organization_db_path,
    resolve_relational_db_path,
    resolve_runtime_events_db_path,
    resolve_session_db_path,
    resolve_sqlite_store_paths,
    resolve_task_checkpoints_db_path,
    resolve_task_memory_db_path,
    resolve_trace_db_path,
    resolve_user_profile_db_path,
)

__all__ = [
    "ENV_SQLITE_DATA_DIR",
    "SQLiteIntegrationBundle",
    "SQLiteIntegrationConfig",
    "SQLiteRelationalStore",
    "SqliteStorePaths",
    "DEFAULT_EXPERIMENTS_DB",
    "DEFAULT_HUMAN_DECISIONS_DB",
    "DEFAULT_RUNTIME_EVENTS_DB",
    "DEFAULT_TASK_CHECKPOINTS_DB",
    "DEFAULT_TASK_MEMORY_DB",
    "DEFAULT_TRACE_DB",
    "ENV_EXPERIMENTS_DB",
    "ENV_HUMAN_DECISIONS_DB",
    "ENV_IDEMPOTENCY_DB",
    "ENV_ORGANIZATION_DB",
    "ENV_RELATIONAL_DB",
    "ENV_RUNTIME_EVENTS_DB",
    "ENV_SESSION_DB",
    "ENV_TASK_CHECKPOINTS_DB",
    "ENV_TASK_MEMORY_DB",
    "ENV_TRACE_DB",
    "ENV_USER_PROFILE_DB",
    "create_sqlite_experiment_store",
    "create_sqlite_human_decision_store",
    "create_sqlite_idempotency_store",
    "create_sqlite_integration",
    "create_sqlite_organization_profile_store",
    "create_sqlite_relational_store",
    "create_sqlite_runtime_event_store",
    "create_sqlite_session_storage",
    "create_sqlite_task_checkpoint_store",
    "create_sqlite_task_memory_store",
    "create_sqlite_trace_store",
    "create_sqlite_user_profile_store",
    "register_sqlite_integration",
    "resolve_experiments_db_path",
    "resolve_human_decisions_db_path",
    "resolve_idempotency_db_path",
    "resolve_organization_db_path",
    "resolve_relational_db_path",
    "resolve_runtime_events_db_path",
    "resolve_session_db_path",
    "resolve_sqlite_config",
    "resolve_sqlite_store_paths",
    "resolve_task_checkpoints_db_path",
    "resolve_task_memory_db_path",
    "resolve_trace_db_path",
    "resolve_user_profile_db_path",
    "create_sqlite_relational_store_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "SQLiteIntegrationBundle",
        "create_sqlite_experiment_store",
        "create_sqlite_human_decision_store",
        "create_sqlite_idempotency_store",
        "create_sqlite_integration",
        "create_sqlite_organization_profile_store",
        "create_sqlite_relational_store",
        "create_sqlite_runtime_event_store",
        "create_sqlite_session_storage",
        "create_sqlite_task_checkpoint_store",
        "create_sqlite_task_memory_store",
        "create_sqlite_trace_store",
        "create_sqlite_user_profile_store",
        "resolve_sqlite_config",
        "register_sqlite_integration",
        "create_sqlite_relational_store_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SQLITE_RELATIONAL_STORE_PROVIDER_ID",
        "SqliteRelationalStoreIntegration",
        "SqliteRelationalStoreIntegrationConfig",
        "SqliteRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_sqlite_integration":
        from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration

        return register_sqlite_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.relational_store.sqlite import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.sqlite import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
