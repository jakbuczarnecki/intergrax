# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mysql relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MYSQL_RELATIONAL_STORE_PROVIDER_ID = "mysql"


class MysqlRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Mysql relational store integration."""

    pass


@runtime_checkable
class MysqlRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class MysqlRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Mysql relational store integration.

    The legacy facade (create_mysql_integration) remains separate and backward-compatible.
    """

    config: MysqlRelationalStoreIntegrationConfig = MysqlRelationalStoreIntegrationConfig()
    _client: MysqlRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: MysqlRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> MysqlRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=MYSQL_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Mysql",
            config=MysqlRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> MysqlRelationalStoreClient | None:
        return self._client
