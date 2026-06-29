# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Snowflake relational store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Mapping, Any

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SNOWFLAKE_RELATIONAL_STORE_PROVIDER_ID = "snowflake"


class SnowflakeRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Snowflake relational store integration."""

    pass


SnowflakeRelationalStoreClient = RelationalStore

class SnowflakeRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Single public Snowflake relational store entrypoint.

    Legacy catalog factory (create_snowflake_relational_store) owns catalog behavior; legacy factories use from_client().
    """

    config: SnowflakeRelationalStoreIntegrationConfig = SnowflakeRelationalStoreIntegrationConfig()
    _client: SnowflakeRelationalStoreClient | None = PrivateAttr(default=None)
    


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
        client: SnowflakeRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> SnowflakeRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=SNOWFLAKE_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Snowflake",
            config=SnowflakeRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SnowflakeRelationalStoreClient | None:
        return self._client

RelationalStore.register(SnowflakeRelationalStoreIntegration)
