# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Motherduck relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MOTHERDUCK_RELATIONAL_STORE_PROVIDER_ID = "motherduck"


class MotherduckRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Motherduck relational store integration."""

    pass


@runtime_checkable
class MotherduckRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class MotherduckRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Motherduck relational store integration.

    The legacy facade (create_motherduck_relational_store) remains separate and backward-compatible.
    """

    config: MotherduckRelationalStoreIntegrationConfig = MotherduckRelationalStoreIntegrationConfig()
    _client: MotherduckRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: MotherduckRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> MotherduckRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=MOTHERDUCK_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Motherduck",
            config=MotherduckRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> MotherduckRelationalStoreClient | None:
        return self._client
