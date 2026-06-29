# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete MySQL integration bundle — the single composition root for MySQL in Intergrax.

Connections are opened only in ``opens.py`` (``pymysql.connect``). Tier-3 code MUST use
``create_mysql_relational_store()``, ``create_mysql_integration()``, or
``profile.resolve(IntegrationCategory.RELATIONAL_STORE)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.providers.relational_store.mysql.adapter import _MySQLRelationalStore
from intergrax.integrations.providers.relational_store.mysql.config import MySQLIntegrationConfig
from intergrax.integrations.providers.relational_store.mysql.opens import open_mysql_relational_store


@dataclass(frozen=True)
class MySQLIntegrationBundle:
    config: MySQLIntegrationConfig
    relational_store: MysqlRelationalStoreIntegration


def resolve_mysql_config(**overrides: object) -> MySQLIntegrationConfig:
    return MySQLIntegrationConfig.from_env(**overrides)


def create_mysql_integration(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> MySQLIntegrationBundle:
    config = resolve_mysql_config(**config_overrides)
    store = open_mysql_relational_store(
        config,
        implementation=relational_store,
        connection_factory=connection_factory,
    )
    assert isinstance(store, MysqlRelationalStoreIntegration)
    return MySQLIntegrationBundle(config=config, relational_store=store)


def create_mysql_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> MysqlRelationalStoreIntegration:
    """Catalog factory for ``"mysql"`` / ``RELATIONAL_STORE``."""
    bundle = create_mysql_integration(
        relational_store=relational_store,
        connection_factory=connection_factory,
        **config_overrides,
    )
    return bundle.relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.mysql.integration import (
    MYSQL_RELATIONAL_STORE_PROVIDER_ID,
    MysqlRelationalStoreIntegration,
    MysqlRelationalStoreIntegrationConfig,
    MysqlRelationalStoreClient,
)


def create_mysql_relational_store_integration(
    *,
    client: MysqlRelationalStoreClient | None = None,
    enabled: bool = False,
) -> MysqlRelationalStoreIntegration:
    """
    Build a contract-based Mysql relational store integration.

    Compatibility shim — constructs Integration via from_store (create_mysql_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Mysql relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return MysqlRelationalStoreIntegration.from_client(client, enabled=enabled)
    return MysqlRelationalStoreIntegration.for_provider(
        provider_id=MYSQL_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Mysql",
        config=MysqlRelationalStoreIntegrationConfig(enabled=enabled),
    )
