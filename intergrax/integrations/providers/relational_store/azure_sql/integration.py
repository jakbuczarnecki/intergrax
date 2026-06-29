# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Sql relational store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Mapping, Any

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID = "azure_sql"


class AzureSqlRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure Sql relational store integration."""

    pass


AzureSqlRelationalStoreClient = RelationalStore

class AzureSqlRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Single public Azure Sql relational store entrypoint.

    Legacy catalog factory (create_azure_sql_relational_store) owns catalog behavior; legacy factories use from_client().
    """

    config: AzureSqlRelationalStoreIntegrationConfig = AzureSqlRelationalStoreIntegrationConfig()
    _client: AzureSqlRelationalStoreClient | None = PrivateAttr(default=None)
    


    def connect(self) -> None:
        self._require_client().connect()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self._require_client().execute(sql, params)

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        return self._require_client().fetch_all(sql, params)

    def close(self) -> None:
        self._require_client().close()


    def _require_client(self) -> RelationalStore:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


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

RelationalStore.register(AzureSqlRelationalStoreIntegration)
