# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration resolution for Tier-3 composition (Phase M.3)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from intergrax.integrations._shared.config import merge_config, read_integration_slug_from_env
from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationCategoryMismatchError,
    IntegrationConfigurationError,
    UnknownIntegrationCategoryError,
    normalize_category,
)
from intergrax.integrations.registry.catalog import get_entry
from intergrax.integrations.registry.profile import IntegrationProfile, default_lab_profile
from intergrax.integrations.registry.slugs import IntegrationSlug, SlugInput, coerce_slug


def build_profile_from_mapping(data: Mapping[str, Any]) -> IntegrationProfile:
    integrations = data.get("integrations", data)
    if not isinstance(integrations, Mapping):
        raise TypeError("integration profile mapping must be a dict")
    return IntegrationProfile.model_validate(dict(integrations))


def build_profile_from_env(
    *,
    defaults: Optional[IntegrationProfile] = None,
) -> IntegrationProfile:
    """
    Build profile from ``INTERGRAX_INTEGRATION_<CATEGORY>`` env vars.

    Unset categories inherit from ``defaults`` when provided.
    """
    base = defaults or IntegrationProfile()
    payload: dict[str, Any] = base.model_dump()

    for category in IntegrationCategory:
        env_slug = read_integration_slug_from_env(category.value)
        if env_slug:
            field = category.value
            if field in IntegrationProfile.model_fields:
                payload[field] = env_slug

    return IntegrationProfile.model_validate(payload)


def resolve_slug(
    category: str | IntegrationCategory,
    *,
    slug: SlugInput | None = None,
    profile: Optional[IntegrationProfile] = None,
) -> str:
    normalized = normalize_category(category)
    if slug is not None:
        return coerce_slug(slug).value

    if profile is not None:
        from_profile = profile.slug_for_category(normalized)
        if from_profile:
            return from_profile.strip().lower()

    from_env = read_integration_slug_from_env(normalized.value)
    if from_env:
        return from_env.strip().lower()

    raise IntegrationConfigurationError(
        f"No integration slug configured for category '{normalized.value}'."
    )


def resolve(
    category: str | IntegrationCategory,
    slug: SlugInput | None = None,
    *,
    profile: Optional[IntegrationProfile] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    Instantiate the provider for ``category``.

    Resolution order: explicit ``slug`` → ``profile`` → env → error.
    """
    normalized = normalize_category(category)
    resolved_slug = resolve_slug(normalized, slug=slug, profile=profile)
    entry = get_entry(resolved_slug)

    if normalized not in entry.categories:
        raise IntegrationCategoryMismatchError(resolved_slug, normalized.value)

    options: dict[str, Any] = {}
    if profile is not None:
        options = profile.options_for_slug(resolved_slug)
    merged = merge_config(options, config)

    if merged:
        return entry.factory(**merged)
    return entry.factory()


def resolve_from_profile(
    profile: IntegrationProfile,
    category: str | IntegrationCategory,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Any:
    return resolve(category, profile=profile, config=config)
