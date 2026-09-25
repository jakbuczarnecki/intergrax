# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
SQLite runtime persistence composition — platform composition root for domain stores.

Uses SQLite provider primitives (config, paths, relational integration) and constructs
domain-owned store implementations through existing owner factories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.experiments.persistence_contract import ExperimentPersistence
from intergrax.integrations.providers.relational_store.sqlite.adapter import (
    _SQLiteRelationalStore,
)
from intergrax.integrations.providers.relational_store.sqlite.config import (
    SQLiteIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.sqlite.integration import (
    SqliteRelationalStoreIntegration,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    SqliteStorePaths,
    ensure_parent_dirs,
    resolve_sqlite_store_paths,
)
from intergrax.memory.contracts.session_storage import SessionStorage
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.human.persistence_contract import HumanDecisionPersistence
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceStore
from intergrax.runtime.organization.organization_profile_store import (
    OrganizationProfileStore,
)
from intergrax.runtime.persistence.sqlite_opens import (
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
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence


@dataclass(frozen=True)
class SQLiteRuntimePersistenceBundle:
    """SQLite-backed runtime persistence facades composed at the platform layer."""

    config: SQLiteIntegrationConfig
    paths: SqliteStorePaths
    relational_store: SqliteRelationalStoreIntegration
    trace_store: RunTraceStore
    runtime_event_store: RuntimeEventPersistence
    task_checkpoint_store: TaskCheckpointPersistence
    human_decision_store: HumanDecisionPersistence
    task_memory_store: TaskMemoryPersistence
    experiment_store: ExperimentPersistence
    idempotency_store: IdempotencyStore
    session_storage: SessionStorage
    organization_profile_store: OrganizationProfileStore
    user_profile_store: UserProfileStore


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


def create_sqlite_runtime_persistence(
    *,
    data_dir: Path | str | None = None,
    **config_overrides: object,
) -> SQLiteRuntimePersistenceBundle:
    """Single entry point for SQLite runtime persistence composition."""
    config, paths = _build_paths(data_dir=data_dir, **config_overrides)

    adapter = _SQLiteRelationalStore(paths.relational)
    relational = SqliteRelationalStoreIntegration.from_client(adapter)
    relational.connect()

    return SQLiteRuntimePersistenceBundle(
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
        organization_profile_store=open_organization_profile_store_at(
            paths.organization
        ),
        user_profile_store=open_user_profile_store_at(paths.user_profile),
    )


def create_sqlite_trace_store(
    *,
    data_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    **config_overrides: object,
) -> RunTraceStore:
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
) -> RuntimeEventPersistence:
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
) -> TaskCheckpointPersistence:
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
) -> HumanDecisionPersistence:
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
) -> TaskMemoryPersistence:
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
) -> ExperimentPersistence:
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
) -> IdempotencyStore:
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
) -> SessionStorage:
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
) -> OrganizationProfileStore:
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
) -> UserProfileStore:
    overrides: dict[str, object] = dict(config_overrides)
    if db_path is not None:
        overrides["user_profile_db"] = Path(db_path)
    _, paths = _build_paths(data_dir=data_dir, **overrides)
    return open_user_profile_store_at(paths.user_profile)


__all__ = [
    "SQLiteRuntimePersistenceBundle",
    "create_sqlite_experiment_store",
    "create_sqlite_human_decision_store",
    "create_sqlite_idempotency_store",
    "create_sqlite_organization_profile_store",
    "create_sqlite_runtime_event_store",
    "create_sqlite_runtime_persistence",
    "create_sqlite_session_storage",
    "create_sqlite_task_checkpoint_store",
    "create_sqlite_task_memory_store",
    "create_sqlite_trace_store",
    "create_sqlite_user_profile_store",
    "resolve_sqlite_config",
]
