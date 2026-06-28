# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Brave search provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BRAVE_SEARCH_PROVIDER_PROVIDER_ID = "brave"


class BraveSearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Brave search provider integration."""

    pass


@runtime_checkable
class BraveSearchProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class BraveSearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Brave search provider integration.

    The legacy facade (create_brave_search_provider) remains separate and backward-compatible.
    """

    config: BraveSearchProviderIntegrationConfig = BraveSearchProviderIntegrationConfig()
    _client: BraveSearchProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: BraveSearchProviderClient,
        *,
        enabled: bool = False,
    ) -> BraveSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=BRAVE_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Brave",
            config=BraveSearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> BraveSearchProviderClient | None:
        return self._client
