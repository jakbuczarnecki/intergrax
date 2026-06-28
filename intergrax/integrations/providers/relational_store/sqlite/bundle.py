# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete SQLite integration bundle — the single composition root for SQLite in Intergrax.

All runtime wiring (trace, events, checkpoints, HITL, task memory, experiments,
idempotency, session, organization profile) MUST use this module or
``profile.resolve(IntegrationCategory.RELATIONAL_STORE)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from intergrax.integrations.providers.relational_store.sqlite.adapter import SQLiteRelationalStore
from intergrax.integrations.providers.relational_store.sqlite.config import SQLiteIntegrationConfig
from intergrax.integrations.providers.relational_store.sqlite.opens import (
    open_experiment_store_at,
    open_human_decision_store_at,
    open_idempotency_store_at,
    open_organization_profile_store_at,
    open_runtime_event_store_at,
    open_session_storage_at,
    open_task_checkpoint_store_at,
    open_task_memory_store_at,
    open_trace_store_at,
    open_user_profile_store_at,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    SqliteStorePaths,
    ensure_parent_dirs,
    resolve_sqlite_store_paths,
)
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore


@dataclass(frozen=True)
class SQLiteIntegrationBundle:
    """All SQLite-backed Tier-0 facades for lab / local deployments."""

    config: SQLiteIntegrationConfig
    paths: SqliteStorePaths
    relational_store: SQLiteRelationalStore
    trace_store: SQLiteRunTraceStore
    runtime_event_store: SQLiteRuntimeEventStore
    task_checkpoint_store: SQLiteTaskCheckpointStore
    human_decision_store: object
    task_memory_store: object
    experiment_store: object
    idempotency_store: object
    session_storage: object
    organization_profile_store: object
    user_profile_store: object


def resolve_sqlite_config(**overrides: object) -> SQLiteIntegrationConfig:
    return SQLiteIntegrationConfig.from_env(**overrides)


def _build_paths(
    *,
    data_dir: Path | str | None = None,
    **config_overrides: object,
) -> tuple[SQLiteIntegrationConfig, SqliteStorePaths]:
    overrides: dict[str, object] = dict(config_overrides)
    if data_dir is not None:
        overrides["data_dir"] = Path(data_dir)
    config = resolve_sqlite_config(**overrides)
    paths = resolve_sqlite_store_paths(config)
    ensure_parent_dirs(paths)
    return config, paths


def create_sqlite_integration(
    *,
    data_dir: Path | str | None = None,
    **config_overrides: object,
) -> SQLiteIntegrationBundle:
    """Single entry point for SQLite — paths and all domain store facades."""
    config, paths = _build_paths(data_dir=data_dir, **config_overrides)

    relational = SQLiteRelationalStore(paths.relational)
    relational.connect()

    return SQLiteIntegrationBundle(
        config=config,
        paths=paths,
        relational_store=relational,
        trace_store=open_trace_store_at(paths.trace),
        runtime_event_store=open_runtime_event_store_at(paths.runtime_events),
        task_checkpoint_store=open_task_checkpoint_store_at(paths.task_checkpoints),
        human_decision_store=open_human_decision_store_at(paths.human_decisions),
        task_memory_store=open_task_memory_store_at(paths.task_memory),
        experiment_store=open_experiment_store_at(paths.experiments),
        idempotency_store=open_idempotency_store_at(paths.idempotency),
        session_storage=open_session_storage_at(paths.session),
        organization_profile_store=open_organization_profile_store_at(paths.organization),
        user_profile_store=open_user_profile_store_at(paths.user_profile),
    )


def create_sqlite_relational_store(
    *,
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    **config_overrides: object,
) -> SQLiteRelationalStore:
    """Catalog factory for ``"sqlite"`` / ``RELATIONAL_STORE``."""
    overrides: dict[str, object] = dict(config_overrides)
    if db_path is not None:
        overrides["relational_db"] = Path(db_path)
    _, paths = _build_paths(data_dir=data_dir, **overrides)
    store = SQLiteRelationalStore(paths.relational)
    store.connect()
    return store


def create_sqlite_trace_store(
    *,
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    **config_overrides: object,
) -> object:
    overrides: dict[str, object] = dict(config_overrides)
    if db_path is not None:
        overrides["trace_db"] = Path(db_path)
    _, paths = _build_paths(data_dir=data_dir, **overrides)
    return open_trace_store_at(paths.trace)


def create_sqlite_runtime_event_store(
    *,
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    **config_overrides: object,
) -> object:
    overrides: dict[str, object] = dict(config_overrides)
    if db_path is not None:
        overrides["runtime_events_db"] = Path(db_path)
    _, paths = _build_paths(data_dir=data_dir, **overrides)
    return open_runtime_event_store_at(paths.runtime_events)


def create_sqlite_task_checkpoint_store(
    *,
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    **config_overrides: object,
) -> object:
    overrides: dict[str, object] = dict(config_overrides)
    if db_path is not None:
        overrides["task_checkpoints_db"] = Path(db_path)
    _, paths = _build_paths(data_dir=data_dir, **overrides)
    return open_task_checkpoint_store_at(paths.task_checkpoints)


def create_sqlite_human_decision_store(
    *,
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    **config_overrides: object,
) -> object:
    overrides: dict[str, object] = dict(config_overrides)
    if db_path is not None:
        overrides["human_decisions_db"] = Path(db_path)
    _, paths = _build_paths(data_dir=data_dir, **overrides)
    return open_human_decision_store_at(paths.human_decisions)


def create_sqlite_task_memory_store(
    *,
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    **config_overrides: object,
) -> object:
    overrides: dict[str, object] = dict(config_overrides)
    if db_path is not None:
        overrides["task_memory_db"] = Path(db_path)
    _, paths = _build_paths(data_dir=data_dir, **overrides)
    return open_task_memory_store_at(paths.task_memory)


def create_sqlite_experiment_store(
    *,
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    **config_overrides: object,
) -> object:
    overrides: dict[str, object] = dict(config_overrides)
    if db_path is not None:
        overrides["experiments_db"] = Path(db_path)
    _, paths = _build_paths(data_dir=data_dir, **overrides)
    return open_experiment_store_at(paths.experiments)


def create_sqlite_idempotency_store(
    *,
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    **config_overrides: object,
) -> object:
    overrides: dict[str, object] = dict(config_overrides)
    if db_path is not None:
        overrides["idempotency_db"] = Path(db_path)
    _, paths = _build_paths(data_dir=data_dir, **overrides)
    return open_idempotency_store_at(paths.idempotency)


def create_sqlite_session_storage(
    *,
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    **config_overrides: object,
) -> object:
    overrides: dict[str, object] = dict(config_overrides)
    if db_path is not None:
        overrides["session_db"] = Path(db_path)
    _, paths = _build_paths(data_dir=data_dir, **overrides)
    return open_session_storage_at(paths.session)


def create_sqlite_organization_profile_store(
    *,
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    **config_overrides: object,
) -> object:
    overrides: dict[str, object] = dict(config_overrides)
    if db_path is not None:
        overrides["organization_db"] = Path(db_path)
    _, paths = _build_paths(data_dir=data_dir, **overrides)
    return open_organization_profile_store_at(paths.organization)


def create_sqlite_user_profile_store(
    *,
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    **config_overrides: object,
) -> object:
    overrides: dict[str, object] = dict(config_overrides)
    if db_path is not None:
        overrides["user_profile_db"] = Path(db_path)
    _, paths = _build_paths(data_dir=data_dir, **overrides)
    return open_user_profile_store_at(paths.user_profile)

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.sqlite.integration import (
    SQLITE_RELATIONAL_STORE_PROVIDER_ID,
    SqliteRelationalStoreIntegration,
    SqliteRelationalStoreIntegrationConfig,
    SqliteRelationalStoreClient,
)


def create_sqlite_relational_store_integration(
    *,
    client: SqliteRelationalStoreClient | None = None,
    enabled: bool = False,
) -> SqliteRelationalStoreIntegration:
    """
    Build a contract-based Sqlite relational store integration.

    The legacy facade (create_sqlite_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Sqlite relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SqliteRelationalStoreIntegration.from_client(client, enabled=enabled)
    return SqliteRelationalStoreIntegration.for_provider(
        provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Sqlite",
        config=SqliteRelationalStoreIntegrationConfig(enabled=enabled),
    )
