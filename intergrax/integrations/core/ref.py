# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolve integration references to catalog slugs (open registry, no enum gate)."""

from __future__ import annotations

from typing import Any, Union

from intergrax.integrations.contracts.base import (
    PROFILE_FIELD_BY_CATEGORY,
    IntegrationCategory,
    UnknownIntegrationError,
)
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.core.plugin import IntegrationPlugin, integration_manifest_for_plugin

# Authoring input for IntegrationProfile fields (before normalization to IntegrationBinding).
IntegrationRef = Union[
    IntegrationManifest,
    type[IntegrationPlugin],
    IntegrationBinding,
    str,
    Any,
]


def _category_for_profile_field(field_name: str) -> IntegrationCategory:
    for category_value, profile_field in PROFILE_FIELD_BY_CATEGORY.items():
        if profile_field == field_name:
            return IntegrationCategory(category_value)
    raise ValueError(f"Unknown integration profile field: {field_name!r}")


def is_integration_instance(value: object) -> bool:
    """True when value is a live integration object (not a manifest, plugin type, or slug)."""
    if value is None:
        return False
    if isinstance(value, (IntegrationBinding, IntegrationManifest)):
        return False
    if isinstance(value, type):
        return False
    if isinstance(value, str):
        return False
    return True


def normalize_integration_binding(value: IntegrationRef | None) -> IntegrationBinding | None:
    if value is None or value == "":
        return None
    if isinstance(value, IntegrationBinding):
        return value
    if isinstance(value, dict):
        binding = IntegrationBinding.model_validate(value)
        if binding.instance is not None:
            return binding
        slug = binding.resolved_slug()
        if slug:
            return IntegrationBinding.from_slug(slug)
        return binding
    if isinstance(value, IntegrationManifest):
        return IntegrationBinding.from_manifest(value)
    if isinstance(value, type) and issubclass(value, IntegrationPlugin):
        return IntegrationBinding.from_plugin(value)
    if is_integration_instance(value):
        return IntegrationBinding.from_instance(value)

    if isinstance(value, str):
        return IntegrationBinding.from_slug(value)
    raise TypeError(
        f"Unsupported integration reference {type(value)!r}; "
        "use IntegrationManifest, IntegrationPlugin class, IntegrationBinding, slug str, or instance."
    )


def resolve_ref_to_slug(
    value: IntegrationRef,
    *,
    field_name: str | None = None,
) -> str:
    binding = normalize_integration_binding(value)
    if binding is None:
        raise ValueError("integration reference is empty")
    if binding.instance is not None:
        raise ValueError(
            "Cannot resolve slug from a pre-built integration instance; use IntegrationBinding.instance at resolve time."
        )
    slug = binding.resolved_slug()
    if not slug:
        raise ValueError("integration reference has no slug")
    normalized = slug.strip().lower()
    manifest = binding.catalog_manifest()
    if field_name is not None:
        category = _category_for_profile_field(field_name)
        if manifest is not None and manifest.categories:
            if category not in manifest.categories:
                allowed = ", ".join(c.value for c in manifest.categories)
                raise ValueError(
                    f"Integration {normalized!r} is not valid for profile field {field_name!r} "
                    f"(category {category.value}); manifest declares: {allowed}"
                )

    from intergrax.integrations.registry.catalog import get_entry

    try:
        entry = get_entry(normalized)
    except UnknownIntegrationError as exc:
        if manifest is not None and manifest.categories:
            return normalized
        known = ", ".join(sorted(_known_slugs_hint()))
        raise ValueError(
            f"Unknown integration slug {normalized!r}. Register the provider first. Known: {known}"
        ) from exc

    if field_name is not None:
        category = _category_for_profile_field(field_name)
        if category not in entry.categories:
            allowed = ", ".join(c.value for c in entry.categories)
            raise ValueError(
                f"Integration {normalized!r} is not valid for profile field {field_name!r} "
                f"(category {category.value}); registered for: {allowed}"
            )
    return normalized


def validate_integration_ref(
    field_name: str,
    value: IntegrationRef | None,
) -> IntegrationBinding | None:
    if value is None or value == "":
        return None
    binding = normalize_integration_binding(value)
    assert binding is not None
    if binding.instance is not None:
        return binding
    resolve_ref_to_slug(binding, field_name=field_name)
    return binding


def _known_slugs_hint() -> list[str]:
    from intergrax.integrations.registry.catalog import list_slugs

    try:
        return list_slugs()
    except Exception:
        return []
