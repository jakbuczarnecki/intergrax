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
from intergrax.integrations.contracts.binding import IntegrationBinding
from intergrax.integrations.contracts.manifest import IntegrationManifest
from intergrax.integrations.contracts.plugin import IntegrationPlugin, integration_manifest_for_plugin

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


