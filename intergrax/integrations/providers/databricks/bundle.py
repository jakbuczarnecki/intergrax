# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Databricks integration bundle — the single composition root for Databricks in Intergrax.

Connections are opened only in ``opens.py`` (``databricks.sql.connect``). Tier-3 code MUST use
``create_databricks_relational_store()``, ``create_databricks_integration()``, or
``profile.resolve(IntegrationCategory.RELATIONAL_STORE)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.providers.databricks.adapter import DatabricksRelationalStore
from intergrax.integrations.providers.databricks.config import DatabricksIntegrationConfig
from intergrax.integrations.providers.databricks.opens import open_databricks_relational_store


@dataclass(frozen=True)
class DatabricksIntegrationBundle:
    config: DatabricksIntegrationConfig
    relational_store: DatabricksRelationalStore


def resolve_databricks_config(**overrides: object) -> DatabricksIntegrationConfig:
    return DatabricksIntegrationConfig.from_env(**overrides)


def create_databricks_integration(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> DatabricksIntegrationBundle:
    config = resolve_databricks_config(**config_overrides)
    store = open_databricks_relational_store(
        config,
        implementation=relational_store,
        connection_factory=connection_factory,
    )
    assert isinstance(store, DatabricksRelationalStore)
    return DatabricksIntegrationBundle(config=config, relational_store=store)


def create_databricks_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> DatabricksRelationalStore:
    """Catalog factory for ``IntegrationSlug.DATABRICKS`` / ``RELATIONAL_STORE``."""
    bundle = create_databricks_integration(
        relational_store=relational_store,
        connection_factory=connection_factory,
        **config_overrides,
    )
    return bundle.relational_store
