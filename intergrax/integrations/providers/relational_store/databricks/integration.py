# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Databricks relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DATABRICKS_RELATIONAL_STORE_PROVIDER_ID = "databricks"


class DatabricksRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Databricks relational store integration."""

    pass


@runtime_checkable
class DatabricksRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class DatabricksRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Databricks relational store integration.

    The legacy facade (create_databricks_integration) remains separate and backward-compatible.
    """

    config: DatabricksRelationalStoreIntegrationConfig = DatabricksRelationalStoreIntegrationConfig()
    _client: DatabricksRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: DatabricksRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> DatabricksRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Databricks",
            config=DatabricksRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> DatabricksRelationalStoreClient | None:
        return self._client
