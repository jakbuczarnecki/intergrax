# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolve SQLite file paths for all runtime domain stores."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from intergrax.integrations.providers.relational_store.sqlite.config import DEFAULT_DATA_DIR, SQLiteIntegrationConfig

ENV_TRACE_DB = "INTERGRAX_TRACE_DB"
ENV_RUNTIME_EVENTS_DB = "INTERGRAX_RUNTIME_EVENTS_DB"
ENV_TASK_CHECKPOINTS_DB = "INTERGRAX_TASK_CHECKPOINTS_DB"
ENV_HUMAN_DECISIONS_DB = "INTERGRAX_HUMAN_DECISIONS_DB"
ENV_TASK_MEMORY_DB = "INTERGRAX_TASK_MEMORY_DB"
ENV_EXPERIMENTS_DB = "INTERGRAX_EXPERIMENTS_DB"
ENV_IDEMPOTENCY_DB = "INTERGRAX_IDEMPOTENCY_DB"
ENV_SESSION_DB = "INTERGRAX_SESSION_DB"
ENV_ORGANIZATION_DB = "INTERGRAX_ORGANIZATION_DB"
ENV_USER_PROFILE_DB = "INTERGRAX_USER_PROFILE_DB"
ENV_RELATIONAL_DB = "INTERGRAX_RELATIONAL_DB"

RELATIONAL_DB_NAME = "intergrax.db"
TRACE_DB_NAME = "intergrax_trace.db"
RUNTIME_EVENTS_DB_NAME = "intergrax_runtime_events.db"
TASK_CHECKPOINTS_DB_NAME = "intergrax_task_checkpoints.db"
HUMAN_DECISIONS_DB_NAME = "intergrax_human_decisions.db"
TASK_MEMORY_DB_NAME = "intergrax_task_memory.db"
EXPERIMENTS_DB_NAME = "intergrax_experiments.db"
IDEMPOTENCY_DB_NAME = "intergrax_idempotency.db"
SESSION_DB_NAME = "intergrax_session.db"
ORGANIZATION_DB_NAME = "intergrax_organization.db"
USER_PROFILE_DB_NAME = "intergrax_user_profile.db"

DEFAULT_TRACE_DB = DEFAULT_DATA_DIR / TRACE_DB_NAME
DEFAULT_RUNTIME_EVENTS_DB = DEFAULT_DATA_DIR / RUNTIME_EVENTS_DB_NAME
DEFAULT_TASK_CHECKPOINTS_DB = DEFAULT_DATA_DIR / TASK_CHECKPOINTS_DB_NAME
DEFAULT_HUMAN_DECISIONS_DB = DEFAULT_DATA_DIR / HUMAN_DECISIONS_DB_NAME
DEFAULT_TASK_MEMORY_DB = DEFAULT_DATA_DIR / TASK_MEMORY_DB_NAME
DEFAULT_EXPERIMENTS_DB = DEFAULT_DATA_DIR / EXPERIMENTS_DB_NAME


@dataclass(frozen=True)
class SqliteStorePaths:
    data_dir: Path
    relational: Path
    trace: Path
    runtime_events: Path
    task_checkpoints: Path
    human_decisions: Path
    task_memory: Path
    experiments: Path
    idempotency: Path
    session: Path
    organization: Path
    user_profile: Path


def _resolve_path(
    *,
    explicit: Path | None,
    env_var: str,
    data_dir: Path,
    default_name: str,
) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get(env_var, "").strip()
    if env:
        return Path(env)
    return data_dir / default_name


def resolve_sqlite_store_paths(config: SQLiteIntegrationConfig) -> SqliteStorePaths:
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    return SqliteStorePaths(
        data_dir=data_dir,
        relational=_resolve_path(
            explicit=config.relational_db,
            env_var=ENV_RELATIONAL_DB,
            data_dir=data_dir,
            default_name=RELATIONAL_DB_NAME,
        ),
        trace=_resolve_path(
            explicit=config.trace_db,
            env_var=ENV_TRACE_DB,
            data_dir=data_dir,
            default_name=TRACE_DB_NAME,
        ),
        runtime_events=_resolve_path(
            explicit=config.runtime_events_db,
            env_var=ENV_RUNTIME_EVENTS_DB,
            data_dir=data_dir,
            default_name=RUNTIME_EVENTS_DB_NAME,
        ),
        task_checkpoints=_resolve_path(
            explicit=config.task_checkpoints_db,
            env_var=ENV_TASK_CHECKPOINTS_DB,
            data_dir=data_dir,
            default_name=TASK_CHECKPOINTS_DB_NAME,
        ),
        human_decisions=_resolve_path(
            explicit=config.human_decisions_db,
            env_var=ENV_HUMAN_DECISIONS_DB,
            data_dir=data_dir,
            default_name=HUMAN_DECISIONS_DB_NAME,
        ),
        task_memory=_resolve_path(
            explicit=config.task_memory_db,
            env_var=ENV_TASK_MEMORY_DB,
            data_dir=data_dir,
            default_name=TASK_MEMORY_DB_NAME,
        ),
        experiments=_resolve_path(
            explicit=config.experiments_db,
            env_var=ENV_EXPERIMENTS_DB,
            data_dir=data_dir,
            default_name=EXPERIMENTS_DB_NAME,
        ),
        idempotency=_resolve_path(
            explicit=config.idempotency_db,
            env_var=ENV_IDEMPOTENCY_DB,
            data_dir=data_dir,
            default_name=IDEMPOTENCY_DB_NAME,
        ),
        session=_resolve_path(
            explicit=config.session_db,
            env_var=ENV_SESSION_DB,
            data_dir=data_dir,
            default_name=SESSION_DB_NAME,
        ),
        organization=_resolve_path(
            explicit=config.organization_db,
            env_var=ENV_ORGANIZATION_DB,
            data_dir=data_dir,
            default_name=ORGANIZATION_DB_NAME,
        ),
        user_profile=_resolve_path(
            explicit=config.user_profile_db,
            env_var=ENV_USER_PROFILE_DB,
            data_dir=data_dir,
            default_name=USER_PROFILE_DB_NAME,
        ),
    )


def _paths(**config_overrides: object) -> SqliteStorePaths:
    config = SQLiteIntegrationConfig.from_env(**config_overrides)
    return resolve_sqlite_store_paths(config)


def resolve_trace_db_path(explicit: Path | str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    return _paths().trace


def resolve_runtime_events_db_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    return _paths().runtime_events


def resolve_task_checkpoints_db_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    return _paths().task_checkpoints


def resolve_human_decisions_db_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    return _paths().human_decisions


def resolve_task_memory_db_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    return _paths().task_memory


def resolve_experiments_db_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    return _paths().experiments


def resolve_idempotency_db_path(explicit: Path | str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    return _paths().idempotency


def resolve_session_db_path(explicit: Path | str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    return _paths().session


def resolve_user_profile_db_path(explicit: Path | str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    return _paths().user_profile


def resolve_organization_db_path(explicit: Path | str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    return _paths().organization


def resolve_relational_db_path(explicit: Path | str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    return _paths().relational


def ensure_parent_dirs(paths: SqliteStorePaths) -> None:
    for path in (
        paths.relational,
        paths.trace,
        paths.runtime_events,
        paths.task_checkpoints,
        paths.human_decisions,
        paths.task_memory,
        paths.experiments,
        paths.idempotency,
        paths.session,
        paths.organization,
        paths.user_profile,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
