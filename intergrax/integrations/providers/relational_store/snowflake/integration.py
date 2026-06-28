# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Snowflake relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SNOWFLAKE_RELATIONAL_STORE_PROVIDER_ID = "snowflake"


class SnowflakeRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Snowflake relational store integration."""

    pass


@runtime_checkable
class SnowflakeRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SnowflakeRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Snowflake relational store integration.

    The legacy facade (create_snowflake_relational_store) remains separate and backward-compatible.
    """

    config: SnowflakeRelationalStoreIntegrationConfig = SnowflakeRelationalStoreIntegrationConfig()
    _client: SnowflakeRelationalStoreClient | None = PrivateAttr(default=None)

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
