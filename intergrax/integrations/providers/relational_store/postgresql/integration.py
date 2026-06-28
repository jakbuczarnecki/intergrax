# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Postgresql relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

POSTGRESQL_RELATIONAL_STORE_PROVIDER_ID = "postgresql"


class PostgresqlRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Postgresql relational store integration."""

    pass


@runtime_checkable
class PostgresqlRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PostgresqlRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Postgresql relational store integration.

    The legacy facade (create_postgresql_integration) remains separate and backward-compatible.
    """

    config: PostgresqlRelationalStoreIntegrationConfig = PostgresqlRelationalStoreIntegrationConfig()
    _client: PostgresqlRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: PostgresqlRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> PostgresqlRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=POSTGRESQL_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Postgresql",
            config=PostgresqlRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PostgresqlRelationalStoreClient | None:
        return self._client
