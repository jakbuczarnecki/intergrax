# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Google CSE integration — single public entry for Google Custom Search.

Implementation lives under ``intergrax.websearch.providers.google_cse_provider``;
compose only through this package.
"""

from intergrax.integrations.providers.google_cse.config import (
    DEFAULT_TIMEOUT_SECONDS,
    ENV_GOOGLE_CSE_API_KEY,
    ENV_GOOGLE_CSE_CX,
    LEGACY_ENV_API_KEY,
    LEGACY_ENV_CX,
    GoogleCSEIntegrationConfig,
)

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "ENV_GOOGLE_CSE_API_KEY",
    "ENV_GOOGLE_CSE_CX",
    "GoogleCSEIntegrationBundle",
    "GoogleCSEIntegrationConfig",
    "GoogleCSESearchProvider",
    "LEGACY_ENV_API_KEY",
    "LEGACY_ENV_CX",
    "create_google_cse_integration",
    "create_google_cse_search_provider",
    "register_google_cse_integration",
    "resolve_google_cse_config",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "GoogleCSEIntegrationBundle",
        "GoogleCSESearchProvider",
        "create_google_cse_integration",
        "create_google_cse_search_provider",
        "resolve_google_cse_config",
    }
)


def __getattr__(name: str):
    if name == "register_google_cse_integration":
        from intergrax.integrations.providers.google_cse.register import register_google_cse_integration

        return register_google_cse_integration
    if name == "GoogleCSESearchProvider":
        from intergrax.integrations.providers.google_cse.adapter import GoogleCSESearchProvider

        return GoogleCSESearchProvider
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.google_cse import bundle as _bundle

        return getattr(_bundle, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
