# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import List, Optional

from intergrax.websearch.providers.base import WebSearchProvider
from intergrax.websearch.providers.google_cse_provider import GoogleCSEProvider
from intergrax.websearch.providers.bing_provider import BingWebProvider

from intergrax.websearch.pipeline.search_and_read import SearchAndReadPipeline


def build_default_pipeline(
    enable_google_cse: bool = True,
    enable_bing_web: bool = True,
    http_rate_per_sec: float = 2.0,
    http_capacity: int = 5,
) -> SearchAndReadPipeline:
    """
    Builds a default SearchAndReadPipeline instance with commonly used providers.

    Providers:
      - GoogleCSEProvider (if enabled and properly configured)
      - BingWebProvider   (if enabled and properly configured)

    Behavior:
      - Providers that fail to initialize (missing API keys, invalid config)
        are silently skipped.
      - If no provider can be constructed, a ValueError is raised.

    Parameters:
      enable_google_cse : toggle Google Custom Search as a provider.
      enable_bing_web   : toggle Bing Web Search v7 as a provider.
      http_rate_per_sec : global HTTP token refill rate for the pipeline.
      http_capacity     : global HTTP token bucket capacity.

    Returns:
      A configured SearchAndReadPipeline instance ready to handle QuerySpec inputs.
    """
    providers: List[WebSearchProvider] = []

    if enable_google_cse:
        try:
            providers.append(GoogleCSEProvider())
        except Exception:
            # Missing configuration or other init error; skip this provider.
            pass

    if enable_bing_web:
        try:
            providers.append(BingWebProvider())
        except Exception:
            # Missing configuration or other init error; skip this provider.
            pass

    if not providers:
        raise ValueError(
            "build_default_pipeline: no providers available. "
            "Check API key configuration or disable providers explicitly."
        )

    pipeline = SearchAndReadPipeline(
        providers=providers,
        http_rate_per_sec=http_rate_per_sec,
        http_capacity=http_capacity,
    )
    return pipeline
