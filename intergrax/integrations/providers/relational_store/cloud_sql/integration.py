# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cloud Sql relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CLOUD_SQL_RELATIONAL_STORE_PROVIDER_ID = "cloud_sql"


class CloudSqlRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Cloud Sql relational store integration."""

    pass


@runtime_checkable
class CloudSqlRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class CloudSqlRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Cloud Sql relational store integration.

    The legacy facade (create_cloud_sql_relational_store) remains separate and backward-compatible.
    """

    config: CloudSqlRelationalStoreIntegrationConfig = CloudSqlRelationalStoreIntegrationConfig()
    _client: CloudSqlRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: CloudSqlRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> CloudSqlRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=CLOUD_SQL_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Cloud Sql",
            config=CloudSqlRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> CloudSqlRelationalStoreClient | None:
        return self._client
