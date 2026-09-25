# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolve rerank scoring via Integration Library (Phase M.7)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.rerank_provider import RerankProvider
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.integrations.core.slug import SlugInput, coerce_slug


def resolve_rerank_provider(
    slug: SlugInput,
    *,
    profile: IntegrationProfile | None = None,
    **config_overrides: object,
) -> RerankProvider:
    register_default_integrations()
    slug_enum = coerce_slug(slug)
    config = dict(config_overrides)
    if profile is not None:
        config = {**profile.options_for_slug(slug_enum), **config}
    provider = resolve(
        IntegrationCategory.RERANK_PROVIDER,
        slug=slug_enum,
        profile=profile,
        config=config,
    )
    if not isinstance(provider, RerankProvider):
        raise TypeError(
            f"Resolved rerank provider for {slug_enum!r} does not implement RerankProvider"
        )
    return provider
