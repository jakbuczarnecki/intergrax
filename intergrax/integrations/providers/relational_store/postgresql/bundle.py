# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete PostgreSQL integration bundle — the single composition root for PostgreSQL in Intergrax.

Connections are opened only in ``session.PostgreSQLConnectionProvider``. Tier-3 code MUST use
``create_postgresql_relational_store()``, ``create_postgresql_integration()``, or
``profile.resolve(IntegrationCategory.RELATIONAL_STORE)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.collaborative_work.materialization_factory import (
    CollaborativeWorkMaterializationBinding,
    CollaborativeWorkPersistenceFactory,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
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


def _postgresql_config_overrides(
    binding: CollaborativeWorkMaterializationBinding,
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if binding.dsn is not None:
        overrides["dsn"] = binding.dsn
    if binding.host is not None:
        overrides["host"] = binding.host
    if binding.port is not None:
        overrides["port"] = binding.port
    if binding.user is not None:
        overrides["user"] = binding.user
    if binding.password is not None:
        overrides["password"] = binding.password
    if binding.database is not None:
        overrides["database"] = binding.database
    if binding.sslmode is not None:
        overrides["sslmode"] = binding.sslmode
    if binding.tenant_schema is not None:
        overrides["tenant_schema"] = binding.tenant_schema
    return overrides


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

    def materialize_collaborative_work_repositories(
        self,
        binding: CollaborativeWorkMaterializationBinding,
    ) -> CollaborativeWorkRepositories:
        from intergrax.collaborative_work.persistence import (
            open_postgresql_collaborative_work_repositories,
        )

        config = resolve_postgresql_config(**_postgresql_config_overrides(binding))
        return open_postgresql_collaborative_work_repositories(
            config=config,
            connection_factory=binding.connection_factory,
        )


create_postgresql_relational_store: PostgreSQLRelationalStoreFactory & CollaborativeWorkPersistenceFactory = (
    PostgreSQLRelationalStoreFactory()
)

from intergrax.integrations.contracts.base import IntegrationConfigurationError
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
