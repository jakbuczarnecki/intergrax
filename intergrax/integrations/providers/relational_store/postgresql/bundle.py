# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete PostgreSQL integration bundle — the single composition root for PostgreSQL in Intergrax.

Connections are opened only in ``session.PostgreSQLConnectionProvider``. Tier-3 code MUST use
``create_postgresql_relational_store()``, ``create_postgresql_integration()``, or
``profile.resolve(IntegrationCategory.RELATIONAL_STORE)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.collaborative_work.materialization_factory import (
    CollaborativeWorkMaterializationBinder,
    CollaborativeWorkPersistenceFactory,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.providers.relational_store.postgresql.config import PostgreSQLIntegrationConfig
from intergrax.integrations.providers.relational_store.postgresql.opens import open_postgresql_relational_store


@dataclass(frozen=True)
class PostgreSQLIntegrationBundle:
    config: PostgreSQLIntegrationConfig
    relational_store: PostgresqlRelationalStoreIntegration


def resolve_postgresql_config(**overrides: object) -> PostgreSQLIntegrationConfig:
    return PostgreSQLIntegrationConfig.from_env(**overrides)


def create_postgresql_integration(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> PostgreSQLIntegrationBundle:
    config = resolve_postgresql_config(**config_overrides)
    store = open_postgresql_relational_store(
        config,
        implementation=relational_store,
        connection_factory=connection_factory,
    )
    assert isinstance(store, PostgresqlRelationalStoreIntegration)
    return PostgreSQLIntegrationBundle(config=config, relational_store=store)


def _postgresql_materialization_inputs(
    options: Mapping[str, Any],
) -> tuple[dict[str, object], Callable[[], object] | None]:
    connection_factory = options.get("connection_factory")
    if "connection_factory" in options and not callable(connection_factory):
        raise IntegrationConfigurationError(
            "PostgreSQL connection_factory must be callable when explicitly provided."
        )
    overrides: dict[str, object] = {
        key: value
        for key, value in options.items()
        if key != "connection_factory" and value is not None
    }
    bound_factory = connection_factory if callable(connection_factory) else None
    return overrides, bound_factory


@dataclass(frozen=True)
class _PostgreSQLCollaborativeWorkMaterializer:
    _config_overrides: dict[str, object]
    _connection_factory: Callable[[], object] | None

    def materialize_collaborative_work_repositories(
        self,
    ) -> CollaborativeWorkRepositories:
        from intergrax.collaborative_work.persistence import (
            open_postgresql_collaborative_work_repositories,
        )

        config = resolve_postgresql_config(**self._config_overrides)
        return open_postgresql_collaborative_work_repositories(
            config=config,
            connection_factory=self._connection_factory,
        )


class PostgreSQLRelationalStoreFactory:
    """Catalog factory for ``"postgresql"`` / ``RELATIONAL_STORE``."""

    def __call__(
        self,
        *,
        relational_store: Optional[RelationalStore] = None,
        connection_factory: Optional[Callable[[], object]] = None,
        **config_overrides: object,
    ) -> PostgresqlRelationalStoreIntegration:
        bundle = create_postgresql_integration(
            relational_store=relational_store,
            connection_factory=connection_factory,
            **config_overrides,
        )
        return bundle.relational_store

    def bind_collaborative_work_materialization(
        self,
        options: Mapping[str, Any],
    ) -> CollaborativeWorkPersistenceFactory:
        overrides, connection_factory = _postgresql_materialization_inputs(options)
        return _PostgreSQLCollaborativeWorkMaterializer(overrides, connection_factory)


create_postgresql_relational_store: (
    PostgreSQLRelationalStoreFactory & CollaborativeWorkMaterializationBinder
) = PostgreSQLRelationalStoreFactory()

from intergrax.integrations.providers.relational_store.postgresql.integration import (
    POSTGRESQL_RELATIONAL_STORE_PROVIDER_ID,
    PostgresqlRelationalStoreIntegration,
    PostgresqlRelationalStoreIntegrationConfig,
    PostgresqlRelationalStoreClient,
)


def create_postgresql_relational_store_integration(
    *,
    client: PostgresqlRelationalStoreClient | None = None,
    enabled: bool = False,
) -> PostgresqlRelationalStoreIntegration:
    """
    Build a contract-based Postgresql relational store integration.

    Compatibility shim — constructs Integration via from_store (create_postgresql_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Postgresql relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return PostgresqlRelationalStoreIntegration.from_client(client, enabled=enabled)
    return PostgresqlRelationalStoreIntegration.for_provider(
        provider_id=POSTGRESQL_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Postgresql",
        config=PostgresqlRelationalStoreIntegrationConfig(enabled=enabled),
    )
