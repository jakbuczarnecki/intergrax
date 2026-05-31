# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolve rerank scoring via Integration Library (Phase M.7)."""

from __future__ import annotations

from typing import List, Optional

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug, SlugInput, coerce_slug


def resolve_rerank_provider(
    slug: SlugInput,
    *,
    profile: IntegrationProfile | None = None,
    **config_overrides: object,
):
    register_default_integrations()
    slug_enum = coerce_slug(slug)
    config = dict(config_overrides)
    if profile is not None:
        config = {**profile.options_for_slug(slug_enum), **config}
    return resolve(
        IntegrationCategory.RERANK_PROVIDER,
        slug=slug_enum,
        profile=profile,
        config=config,
    )


def rerank_scores(
    slug: SlugInput,
    query: str,
    texts: List[str],
    *,
    profile: IntegrationProfile | None = None,
    top_n: Optional[int] = None,
    **config_overrides: object,
) -> List[float]:
    provider = resolve_rerank_provider(slug, profile=profile, **config_overrides)
    if hasattr(provider, "score"):
        return provider.score(query, texts, top_n=top_n)  # type: ignore[call-arg, union-attr]
    raise TypeError(f"Rerank provider for {slug!r} does not expose score()")
