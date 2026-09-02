# © Artur Czarnecki. All rights reserved.

"""Resolve a production SearchProvider for Q3 qualification from environment credentials."""

from __future__ import annotations

import os
from dataclasses import dataclass

from intergrax.integrations._shared.p2.factories import create_brave_search_provider
from intergrax.integrations._shared.p3.factories import (
    create_exa_search_provider,
    create_tavily_search_provider,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.providers.search_provider.google_cse.bundle import (
    create_google_cse_search_provider,
)
from intergrax.websearch.schemas.search_hit import SearchHit


@dataclass(frozen=True, slots=True)
class ResolvedSearchProvider:
    provider_id: str
    provider: SearchProvider


def _env_nonempty(*names: str) -> bool:
    for name in names:
        if os.environ.get(name, "").strip():
            return True
    return False


def _try_tavily() -> ResolvedSearchProvider | None:
    if not _env_nonempty("INTERGRAX_TAVILY_API_KEY", "TAVILY_API_KEY"):
        return None
    try:
        provider = create_tavily_search_provider()
    except (IntegrationConfigurationError, ValueError, OSError):
        return None
    return ResolvedSearchProvider(provider_id="tavily", provider=provider)


def _try_brave() -> ResolvedSearchProvider | None:
    if not _env_nonempty("INTERGRAX_BRAVE_API_KEY", "BRAVE_API_KEY"):
        return None
    try:
        provider = create_brave_search_provider()
    except (IntegrationConfigurationError, ValueError, OSError):
        return None
    return ResolvedSearchProvider(provider_id="brave", provider=provider)


def _try_google_cse() -> ResolvedSearchProvider | None:
    if not _env_nonempty(
        "INTERGRAX_GOOGLE_CSE_API_KEY",
        "GOOGLE_CSE_API_KEY",
    ) or not _env_nonempty("INTERGRAX_GOOGLE_CSE_CX", "GOOGLE_CSE_CX"):
        return None
    try:
        provider = create_google_cse_search_provider()
    except (IntegrationConfigurationError, ValueError, OSError):
        return None
    return ResolvedSearchProvider(provider_id="google_cse", provider=provider)


def _try_exa() -> ResolvedSearchProvider | None:
    if not _env_nonempty("INTERGRAX_EXA_API_KEY", "EXA_API_KEY"):
        return None
    try:
        provider = create_exa_search_provider()
    except (IntegrationConfigurationError, ValueError, OSError):
        return None
    return ResolvedSearchProvider(provider_id="exa", provider=provider)


def resolve_qualification_search_provider() -> ResolvedSearchProvider:
    for resolver in (_try_tavily, _try_brave, _try_google_cse, _try_exa):
        resolved = resolver()
        if resolved is not None:
            return resolved
    raise IntegrationConfigurationError(
        "No search provider credentials configured "
        "(tried tavily, brave, google_cse, exa)",
    )


def preflight_search_provider(
    resolved: ResolvedSearchProvider,
    *,
    query: str = "Python programming language official site",
    limit: int = 3,
) -> tuple[SearchHit, ...]:
    hits = resolved.provider.search(query, limit=limit)
    if not hits:
        raise IntegrationConfigurationError(
            f"Search provider {resolved.provider_id!r} returned no results",
        )
    return tuple(hits)


__all__ = [
    "ResolvedSearchProvider",
    "preflight_search_provider",
    "resolve_qualification_search_provider",
]
