"""Embedding integration bootstrap helpers for VPI proof runs."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, UnknownIntegrationError
from intergrax.integrations.registry.catalog import get_entry, list_slugs


def ensure_embedding_provider_integrations_registered() -> None:
    """Register the HF embedding provider row required for canonical VPI builds."""
    registered = list_slugs(category=IntegrationCategory.EMBEDDING_PROVIDER)
    if "hf" in registered:
        return
    try:
        get_entry("hf")
    except UnknownIntegrationError:
        from intergrax.integrations.providers.embedding_provider.hf.register import (
            register_hf_embedding_provider_integration,
        )

        register_hf_embedding_provider_integration()
