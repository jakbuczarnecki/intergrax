# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolve integration references against the runtime catalog."""

from __future__ import annotations

from intergrax.integrations.contracts.base import (
    UnknownIntegrationError,
)
from intergrax.integrations.contracts.binding import IntegrationBinding
from intergrax.integrations.contracts.ref import (
    IntegrationRef,
    _category_for_profile_field,
    is_integration_instance,
    normalize_integration_binding,
)

__all__ = [
    "IntegrationRef",
    "is_integration_instance",
    "normalize_integration_binding",
    "resolve_ref_to_slug",
    "validate_integration_ref",
]


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
