# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete PostgreSQL integration bundle — the single composition root for PostgreSQL in Intergrax.

Connections are opened only in ``opens.py`` (``psycopg.connect``). Tier-3 code MUST use
``create_postgresql_relational_store()``, ``create_postgresql_integration()``, or
``profile.resolve(IntegrationCategory.RELATIONAL_STORE)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.providers.postgresql.adapter import PostgreSQLRelationalStore
from intergrax.integrations.providers.postgresql.config import PostgreSQLIntegrationConfig
from intergrax.integrations.providers.postgresql.opens import open_postgresql_relational_store


@dataclass(frozen=True)
class PostgreSQLIntegrationBundle:
    config: PostgreSQLIntegrationConfig
    relational_store: PostgreSQLRelationalStore


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
    assert isinstance(store, PostgreSQLRelationalStore)
    return PostgreSQLIntegrationBundle(config=config, relational_store=store)


def create_postgresql_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> PostgreSQLRelationalStore:
    """Catalog factory for ``IntegrationSlug.POSTGRESQL`` / ``RELATIONAL_STORE``."""
    bundle = create_postgresql_integration(
        relational_store=relational_store,
        connection_factory=connection_factory,
        **config_overrides,
    )
    return bundle.relational_store
