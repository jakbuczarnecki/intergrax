# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Exa search provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

EXA_SEARCH_PROVIDER_PROVIDER_ID = "exa"


class ExaSearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Exa search provider integration."""

    pass


@runtime_checkable
class ExaSearchProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ExaSearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Exa search provider integration.

    The legacy facade (create_exa_search_provider) remains separate and backward-compatible.
    """

    config: ExaSearchProviderIntegrationConfig = ExaSearchProviderIntegrationConfig()
    _client: ExaSearchProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ExaSearchProviderClient,
        *,
        enabled: bool = False,
    ) -> ExaSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=EXA_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Exa",
            config=ExaSearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ExaSearchProviderClient | None:
        return self._client
