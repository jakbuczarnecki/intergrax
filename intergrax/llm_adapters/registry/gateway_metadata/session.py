# © Artur Czarnecki. All rights reserved.

"""Session-scoped gateway metadata merge for context-window resolution (M-LLM-X.14.2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.gateway_metadata.openrouter_client import (
    OpenRouterModelMetadataClient,
)

_session_client: OpenRouterModelMetadataClient | None = None


def _provider_slug(provider: LLMProvider | str) -> str:
    if isinstance(provider, LLMProvider):
        return provider.value
    return str(provider or "").strip().lower()


def reset_gateway_metadata_session() -> None:
    """Clear process-wide gateway metadata cache (tests)."""
    global _session_client
    if _session_client is not None:
        _session_client.reset_cache()
    _session_client = None


def _session_client_for_options(options: Mapping[str, Any]) -> OpenRouterModelMetadataClient:
    global _session_client
    ttl_raw = options.get("gateway_metadata_ttl_seconds", 3600)
    ttl_seconds = float(ttl_raw) if ttl_raw is not None else 3600.0
    fetcher = options.get("gateway_metadata_fetcher")
    if _session_client is None or _session_client._ttl_seconds != ttl_seconds:
        _session_client = OpenRouterModelMetadataClient(
            ttl_seconds=ttl_seconds,
            fetcher=fetcher,
        )
    return _session_client


def lookup_gateway_context_window(
    provider: LLMProvider | str,
    model: str,
    profile_options: Mapping[str, Any] | None,
) -> int | None:
    """
    Resolve context window from gateway metadata when ``fetch_gateway_metadata`` is enabled.

    Does not bypass catalog exact/prefix matches (ADR-LLM-002).
    """
    options = dict(profile_options or {})
    if not options.get("fetch_gateway_metadata"):
        return None
    normalized_model = (model or "").strip()
    if not normalized_model:
        return None
    client = _session_client_for_options(options)
    metadata = client.lookup(normalized_model)
    if metadata is None or metadata.context_window_tokens is None:
        return None
    return int(metadata.context_window_tokens)
