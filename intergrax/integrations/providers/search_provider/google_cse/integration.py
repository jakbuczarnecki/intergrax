# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Cse search provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GOOGLE_CSE_SEARCH_PROVIDER_PROVIDER_ID = "google_cse"


class GoogleCseSearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Google Cse search provider integration."""

    pass


@runtime_checkable
class GoogleCseSearchProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GoogleCseSearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Google Cse search provider integration.

    The legacy facade (create_google_cse_integration) remains separate and backward-compatible.
    """

    config: GoogleCseSearchProviderIntegrationConfig = GoogleCseSearchProviderIntegrationConfig()
    _client: GoogleCseSearchProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: GoogleCseSearchProviderClient,
        *,
        enabled: bool = False,
    ) -> GoogleCseSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=GOOGLE_CSE_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Google Cse",
            config=GoogleCseSearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GoogleCseSearchProviderClient | None:
        return self._client
