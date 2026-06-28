# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Places search provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GOOGLE_PLACES_SEARCH_PROVIDER_PROVIDER_ID = "google_places"


class GooglePlacesSearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Google Places search provider integration."""

    pass


@runtime_checkable
class GooglePlacesSearchProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GooglePlacesSearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Google Places search provider integration.

    The legacy facade (create_google_places_search_provider) remains separate and backward-compatible.
    """

    config: GooglePlacesSearchProviderIntegrationConfig = GooglePlacesSearchProviderIntegrationConfig()
    _client: GooglePlacesSearchProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: GooglePlacesSearchProviderClient,
        *,
        enabled: bool = False,
    ) -> GooglePlacesSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=GOOGLE_PLACES_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Google Places",
            config=GooglePlacesSearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GooglePlacesSearchProviderClient | None:
        return self._client
