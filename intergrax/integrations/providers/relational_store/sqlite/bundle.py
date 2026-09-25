# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
SQLite relational-store integration factory — provider-owned composition only.

Runtime persistence (trace, events, checkpoints, memory, session, …) is composed in
``intergrax.runtime.persistence.sqlite_composition``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.sqlite.adapter import (
    _SQLiteRelationalStore,
)
from intergrax.integrations.providers.relational_store.sqlite.config import (
    SQLiteIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.sqlite.integration import (
    SQLITE_RELATIONAL_STORE_PROVIDER_ID,
    SqliteRelationalStoreIntegration,
    SqliteRelationalStoreIntegrationConfig,
    SqliteRelationalStoreClient,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    SqliteStorePaths,
    ensure_parent_dirs,
    resolve_sqlite_store_paths,
)

if TYPE_CHECKING:
    from intergrax.collaborative_work.materialization_factory import (
        CollaborativeWorkMaterializationBinder,
        CollaborativeWorkPersistenceFactory,
    )
    from intergrax.collaborative_work.persistence import (
        CollaborativeWorkMaterializedRepositories,
    )


def resolve_sqlite_config(**overrides: object) -> SQLiteIntegrationConfig:
    return SQLiteIntegrationConfig.from_env(**overrides)


def _configuration_path_from_options(value: object, *, field_name: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, os.PathLike):
        return Path(value)
    raise IntegrationConfigurationError(
        f"SQLite {field_name} must be a path (str, Path, or path-like), "
        f"got {type(value).__name__}",
    )


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


def _sqlite_materialization_paths_from_options(
    options: Mapping[str, object],
) -> SqliteStorePaths:
    overrides: dict[str, object] = dict(options)
    data_dir_raw = overrides.pop("data_dir", None)
    relational_db_raw = overrides.pop("relational_db", None)
    build_data_dir: Path | str | None
    if data_dir_raw is None:
        build_data_dir = None
    elif isinstance(data_dir_raw, (Path, str)):
        build_data_dir = data_dir_raw
    else:
        build_data_dir = _configuration_path_from_options(
            data_dir_raw,
            field_name="data_dir",
        )
    if relational_db_raw is not None:
        overrides["relational_db"] = _configuration_path_from_options(
            relational_db_raw,
            field_name="relational_db",
        )
    _, paths = _build_paths(data_dir=build_data_dir, **overrides)
    return paths


@dataclass(frozen=True)
class _SQLiteCollaborativeWorkMaterializer:
    _paths: SqliteStorePaths

    def materialize_collaborative_work_repositories(
        self,
    ) -> CollaborativeWorkMaterializedRepositories:
        from intergrax.collaborative_work.persistence import (
            open_sqlite_collaborative_work_repositories,
        )

        return open_sqlite_collaborative_work_repositories(str(self._paths.relational))


class SQLiteRelationalStoreFactory:
    """Catalog factory for ``"sqlite"`` / ``RELATIONAL_STORE``."""

    def __call__(
        self,
        *,
        data_dir: Path | str | None = None,
        db_path: Path | str | None = None,
        **config_overrides: object,
    ) -> SqliteRelationalStoreIntegration:
        overrides: dict[str, object] = dict(config_overrides)
        if db_path is not None:
            overrides["relational_db"] = Path(db_path)
        _, paths = _build_paths(data_dir=data_dir, **overrides)
        store = _SQLiteRelationalStore(paths.relational)
        integration = SqliteRelationalStoreIntegration.from_client(store)
        integration.connect()
        return integration

    def bind_collaborative_work_materialization(
        self,
        options: Mapping[str, object],
    ) -> CollaborativeWorkPersistenceFactory:
        paths = _sqlite_materialization_paths_from_options(options)
        return _SQLiteCollaborativeWorkMaterializer(paths)


create_sqlite_relational_store = SQLiteRelationalStoreFactory()

if TYPE_CHECKING:
    _collaborative_work_materialization_binder: CollaborativeWorkMaterializationBinder = create_sqlite_relational_store


def create_sqlite_relational_store_integration(
    *,
    client: SqliteRelationalStoreClient | None = None,
    enabled: bool = False,
) -> SqliteRelationalStoreIntegration:
    """
    Build a contract-based Sqlite relational store integration.

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
