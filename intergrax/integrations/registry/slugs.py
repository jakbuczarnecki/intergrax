# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Integration slug helpers — open catalog only (no central enum).

Prefer:

- :class:`~intergrax.integrations.core.manifest.IntegrationManifest` co-located in each provider
- :func:`~intergrax.integrations.registry.plugin_register.register_from_manifest`
- :func:`~intergrax.integrations.core.slug.coerce_slug` for string/manifest resolution
"""

from __future__ import annotations

from intergrax.integrations.core.defaults import CLOUD_PLATFORM_DEFAULTS
from intergrax.integrations.core.slug import SlugInput, coerce_slug, slug_value

__all__ = [
    "CLOUD_PLATFORM_DEFAULTS",
    "SlugInput",
    "coerce_slug",
    "slug_value",
]
