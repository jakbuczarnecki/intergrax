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
from intergrax.integrations.providers.relational_store.databricks.adapter import _DatabricksRelationalStore
from intergrax.integrations.providers.relational_store.databricks.config import DatabricksIntegrationConfig
from intergrax.integrations.providers.relational_store.databricks.opens import open_databricks_relational_store


@dataclass(frozen=True)
class DatabricksIntegrationBundle:
    config: DatabricksIntegrationConfig
    relational_store: DatabricksRelationalStoreIntegration


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
    assert isinstance(store, DatabricksRelationalStoreIntegration)
    return DatabricksIntegrationBundle(config=config, relational_store=store)


def create_databricks_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> DatabricksRelationalStoreIntegration:
    """Catalog factory for ``"databricks"`` / ``RELATIONAL_STORE``."""
    bundle = create_databricks_integration(
        relational_store=relational_store,
        connection_factory=connection_factory,
        **config_overrides,
    )
    return bundle.relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.databricks.integration import (
    DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
    DatabricksRelationalStoreIntegration,
    DatabricksRelationalStoreIntegrationConfig,
    DatabricksRelationalStoreClient,
)


def create_databricks_relational_store_integration(
    *,
    client: DatabricksRelationalStoreIntegrationClient | None = None,
    enabled: bool = False,
) -> DatabricksRelationalStoreIntegration:
    """
    Build a contract-based Databricks relational store integration.

    Compatibility shim — constructs Integration via from_store (create_databricks_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Databricks relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return DatabricksRelationalStoreIntegration.from_client(client, enabled=enabled)
    return DatabricksRelationalStoreIntegration.for_provider(
        provider_id=DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Databricks",
        config=DatabricksRelationalStoreIntegrationConfig(enabled=enabled),
    )
