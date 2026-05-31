# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Bing openers — internal to the bing integration package.

Only this module may construct ``BingWebProvider`` instances for catalog wiring.
"""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.providers.search_provider.bing.adapter import BingSearchProvider
from intergrax.integrations.providers.search_provider.bing.config import BingIntegrationConfig
from intergrax.websearch.providers.bing_provider import BingWebProvider


def open_bing_web_search_provider(
    config: BingIntegrationConfig,
    *,
    provider: Optional[BingWebProvider] = None,
    session: Optional[object] = None,
) -> BingWebProvider:
    if provider is not None:
        return provider
    timeout = int(config.timeout_seconds or BingWebProvider._DEFAULT_TIMEOUT)
    return BingWebProvider(
        api_key=config.api_key,
        session=session,
        timeout=timeout,
    )


def open_bing_search_provider(
    config: BingIntegrationConfig,
    *,
    provider: Optional[BingWebProvider] = None,
    session: Optional[object] = None,
) -> SearchProvider:
    return BingSearchProvider(
        open_bing_web_search_provider(config, provider=provider, session=session)
    )
