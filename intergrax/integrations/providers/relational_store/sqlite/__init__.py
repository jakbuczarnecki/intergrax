# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
SQLite relational-store integration — provider config, paths, and catalog registration.

Runtime persistence composition lives in ``intergrax.runtime.persistence.sqlite_composition``.
"""

from intergrax.integrations.providers.relational_store.sqlite.adapter import (
    _SQLiteRelationalStore,
)
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    create_sqlite_relational_store,
    create_sqlite_relational_store_integration,
    resolve_sqlite_config,
)
from intergrax.integrations.providers.relational_store.sqlite.config import (
    ENV_SQLITE_DATA_DIR,
    SQLiteIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.sqlite.integration import (
    SQLITE_RELATIONAL_STORE_PROVIDER_ID,
    SqliteRelationalStoreClient,
    SqliteRelationalStoreIntegration,
    SqliteRelationalStoreIntegrationConfig,
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
from intergrax.integrations.providers.relational_store.sqlite.register import (
    register_sqlite_integration,
)

SQLiteRelationalStore = _SQLiteRelationalStore

__all__ = [
    "ENV_SQLITE_DATA_DIR",
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
    "SQLITE_RELATIONAL_STORE_PROVIDER_ID",
    "SqliteRelationalStoreClient",
    "SqliteRelationalStoreIntegration",
    "SqliteRelationalStoreIntegrationConfig",
    "create_sqlite_relational_store",
    "create_sqlite_relational_store_integration",
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
]
