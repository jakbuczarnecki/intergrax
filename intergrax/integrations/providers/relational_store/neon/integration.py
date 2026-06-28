# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neon relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

NEON_RELATIONAL_STORE_PROVIDER_ID = "neon"


class NeonRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Neon relational store integration."""

    pass


@runtime_checkable
class NeonRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class NeonRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Neon relational store integration.

    The legacy facade (create_neon_relational_store) remains separate and backward-compatible.
    """

    config: NeonRelationalStoreIntegrationConfig = NeonRelationalStoreIntegrationConfig()
    _client: NeonRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: NeonRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> NeonRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=NEON_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Neon",
            config=NeonRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> NeonRelationalStoreClient | None:
        return self._client
