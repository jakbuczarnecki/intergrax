# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Google CSE integration bundle — the single composition root for Google search.

All runtime wiring MUST use this module or
``profile.resolve(IntegrationCategory.SEARCH_PROVIDER)`` with ``IntegrationSlug.GOOGLE_CSE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.providers.search_provider.google_cse.config import GoogleCSEIntegrationConfig
from intergrax.integrations.providers.search_provider.google_cse.opens import (
    open_google_cse_search_provider,
    open_google_cse_web_search_provider,
)
from intergrax.integrations.providers.search_provider.google_cse.web_client import GoogleCSEProvider


@dataclass(frozen=True)
class GoogleCSEIntegrationBundle:
    """Google CSE search provider + underlying websearch implementation."""

    config: GoogleCSEIntegrationConfig
    search_provider: SearchProvider
    web_search_provider: GoogleCSEProvider


def resolve_google_cse_config(**overrides: object) -> GoogleCSEIntegrationConfig:
    return GoogleCSEIntegrationConfig.from_env(**overrides)


def create_google_cse_integration(
    *,
    api_key: Optional[str] = None,
    cx: Optional[str] = None,
    provider: Optional[GoogleCSEProvider] = None,
    session: Optional[object] = None,
    **config_overrides: object,
) -> GoogleCSEIntegrationBundle:
    """Single entry point for Google CSE — catalog facade + legacy WebSearchProvider."""
    overrides: dict[str, object] = dict(config_overrides)
    if api_key is not None:
        overrides["api_key"] = api_key
    if cx is not None:
        overrides["cx"] = cx

    config = resolve_google_cse_config(**overrides)
    web = open_google_cse_web_search_provider(config, provider=provider, session=session)
    search = open_google_cse_search_provider(config, provider=web)

    return GoogleCSEIntegrationBundle(
        config=config,
        search_provider=search,
        web_search_provider=web,
    )


def create_google_cse_search_provider(
    *,
    api_key: Optional[str] = None,
    cx: Optional[str] = None,
    provider: Optional[GoogleCSEProvider] = None,
    session: Optional[object] = None,
    **config_overrides: object,
) -> SearchProvider:
    """Catalog factory for ``IntegrationSlug.GOOGLE_CSE`` / ``SEARCH_PROVIDER``."""
    return create_google_cse_integration(
        api_key=api_key,
        cx=cx,
        provider=provider,
        session=session,
        **config_overrides,
    ).search_provider
