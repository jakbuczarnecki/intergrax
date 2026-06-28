# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Bing integration bundle — the single composition root for Bing Web Search.

All runtime wiring MUST use this module or
``profile.resolve(IntegrationCategory.SEARCH_PROVIDER)`` with ``"bing"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.providers.search_provider.bing.config import BingIntegrationConfig
from intergrax.integrations.providers.search_provider.bing.opens import (
    open_bing_search_provider,
    open_bing_web_search_provider,
)
from intergrax.integrations.providers.search_provider.bing.web_client import BingWebProvider


@dataclass(frozen=True)
class BingIntegrationBundle:
    """Bing search provider + underlying websearch implementation."""

    config: BingIntegrationConfig
    search_provider: SearchProvider
    web_search_provider: BingWebProvider


def resolve_bing_config(**overrides: object) -> BingIntegrationConfig:
    return BingIntegrationConfig.from_env(**overrides)


def create_bing_integration(
    *,
    api_key: Optional[str] = None,
    provider: Optional[BingWebProvider] = None,
    session: Optional[object] = None,
    **config_overrides: object,
) -> BingIntegrationBundle:
    """Single entry point for Bing — catalog facade + legacy WebSearchProvider."""
    overrides: dict[str, object] = dict(config_overrides)
    if api_key is not None:
        overrides["api_key"] = api_key

    config = resolve_bing_config(**overrides)
    web = open_bing_web_search_provider(config, provider=provider, session=session)
    search = open_bing_search_provider(config, provider=web)

    return BingIntegrationBundle(
        config=config,
        search_provider=search,
        web_search_provider=web,
    )


def create_bing_search_provider(
    *,
    api_key: Optional[str] = None,
    provider: Optional[BingWebProvider] = None,
    session: Optional[object] = None,
    **config_overrides: object,
) -> SearchProvider:
    """Catalog factory for ``"bing"`` / ``SEARCH_PROVIDER``."""
    return create_bing_integration(
        api_key=api_key,
        provider=provider,
        session=session,
        **config_overrides,
    ).search_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.search_provider.bing.integration import (
    BING_SEARCH_PROVIDER_PROVIDER_ID,
    BingSearchProviderIntegration,
    BingSearchProviderIntegrationConfig,
    BingSearchProviderClient,
)


def create_bing_search_provider_integration(
    *,
    client: BingSearchProviderClient | None = None,
    enabled: bool = False,
) -> BingSearchProviderIntegration:
    """
    Build a contract-based Bing search provider integration.

    The legacy facade (create_bing_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Bing search provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return BingSearchProviderIntegration.from_client(client, enabled=enabled)
    return BingSearchProviderIntegration.for_provider(
        provider_id=BING_SEARCH_PROVIDER_PROVIDER_ID,
        display_name="Bing",
        config=BingSearchProviderIntegrationConfig(enabled=enabled),
    )
