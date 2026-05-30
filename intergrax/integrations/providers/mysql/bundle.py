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
from intergrax.integrations.providers.mysql.adapter import MySQLRelationalStore
from intergrax.integrations.providers.mysql.config import MySQLIntegrationConfig
from intergrax.integrations.providers.mysql.opens import open_mysql_relational_store


@dataclass(frozen=True)
class MySQLIntegrationBundle:
    config: MySQLIntegrationConfig
    relational_store: MySQLRelationalStore


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
    assert isinstance(store, MySQLRelationalStore)
    return MySQLIntegrationBundle(config=config, relational_store=store)


def create_mysql_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> MySQLRelationalStore:
    """Catalog factory for ``IntegrationSlug.MYSQL`` / ``RELATIONAL_STORE``."""
    bundle = create_mysql_integration(
        relational_store=relational_store,
        connection_factory=connection_factory,
        **config_overrides,
    )
    return bundle.relational_store
