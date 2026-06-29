# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Google CSE openers — internal to the google_cse integration package.

Only this module may construct ``GoogleCSEProvider`` instances for catalog wiring.
"""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.providers.search_provider.google_cse.adapter import _GoogleCSESearchProvider
from intergrax.integrations.providers.search_provider.google_cse.integration import GoogleCseSearchProviderIntegration
from intergrax.integrations.providers.search_provider.google_cse.config import GoogleCSEIntegrationConfig
from intergrax.integrations.providers.search_provider.google_cse.web_client import GoogleCSEProvider


def open_google_cse_web_search_provider(
    config: GoogleCSEIntegrationConfig,
    *,
    provider: Optional[GoogleCSEProvider] = None,
    session: Optional[object] = None,
) -> GoogleCSEProvider:
    if provider is not None:
        return provider
    timeout = int(config.timeout_seconds or GoogleCSEProvider._DEFAULT_TIMEOUT)
    return GoogleCSEProvider(
        api_key=config.api_key,
        cx=config.cx,
        session=session,
        timeout=timeout,
    )


def open_google_cse_search_provider(
    config: GoogleCSEIntegrationConfig,
    *,
    provider: Optional[GoogleCSEProvider] = None,
    session: Optional[object] = None,
) -> SearchProvider:
    return GoogleCseSearchProviderIntegration.from_client(_GoogleCSESearchProvider(
        open_google_cse_web_search_provider(config, provider=provider, session=session))
    )
