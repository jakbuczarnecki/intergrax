# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slug resolution for the open integration catalog (manifest / string / plugin)."""

from __future__ import annotations

from typing import Union

from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.core.plugin import IntegrationPlugin

SlugInput = Union[
    IntegrationManifest,
    type[IntegrationPlugin],
    IntegrationBinding,
    str,
]


def coerce_slug(value: SlugInput) -> str:
    """
    Resolve a catalog slug from manifest, plugin type, binding, or string.

    Validates registration in the runtime catalog when the slug is known at resolve time.
    """
    from intergrax.integrations.core.ref import normalize_integration_binding, resolve_ref_to_slug

    binding = normalize_integration_binding(value)
    if binding is None:
        raise ValueError("integration slug reference is empty")
    return resolve_ref_to_slug(binding)


def slug_value(value: SlugInput | None) -> str | None:
    if value is None:
        return None
    return coerce_slug(value)
