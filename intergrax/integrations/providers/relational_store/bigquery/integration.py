# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bigquery relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BIGQUERY_RELATIONAL_STORE_PROVIDER_ID = "bigquery"


class BigqueryRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Bigquery relational store integration."""

    pass


@runtime_checkable
class BigqueryRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class BigqueryRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Bigquery relational store integration.

    The legacy facade (create_bigquery_relational_store) remains separate and backward-compatible.
    """

    config: BigqueryRelationalStoreIntegrationConfig = BigqueryRelationalStoreIntegrationConfig()
    _client: BigqueryRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: BigqueryRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> BigqueryRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=BIGQUERY_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Bigquery",
            config=BigqueryRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> BigqueryRelationalStoreClient | None:
        return self._client
