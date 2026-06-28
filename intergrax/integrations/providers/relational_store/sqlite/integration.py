# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sqlite relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SQLITE_RELATIONAL_STORE_PROVIDER_ID = "sqlite"


class SqliteRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Sqlite relational store integration."""

    pass


@runtime_checkable
class SqliteRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SqliteRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Sqlite relational store integration.

    The legacy facade (create_sqlite_integration) remains separate and backward-compatible.
    """

    config: SqliteRelationalStoreIntegrationConfig = SqliteRelationalStoreIntegrationConfig()
    _client: SqliteRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: SqliteRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> SqliteRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Sqlite",
            config=SqliteRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SqliteRelationalStoreClient | None:
        return self._client
