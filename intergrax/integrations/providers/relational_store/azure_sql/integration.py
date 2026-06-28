# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Sql relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID = "azure_sql"


class AzureSqlRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure Sql relational store integration."""

    pass


@runtime_checkable
class AzureSqlRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AzureSqlRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Azure Sql relational store integration.

    The legacy facade (create_azure_sql_relational_store) remains separate and backward-compatible.
    """

    config: AzureSqlRelationalStoreIntegrationConfig = AzureSqlRelationalStoreIntegrationConfig()
    _client: AzureSqlRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: AzureSqlRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> AzureSqlRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Azure Sql",
            config=AzureSqlRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AzureSqlRelationalStoreClient | None:
        return self._client
