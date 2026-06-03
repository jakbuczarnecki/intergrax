# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register google_places in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.google_places.bundle import create_google_places_search_provider
from intergrax.integrations.providers.search_provider.google_places.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_google_places_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_google_places_search_provider, override=override)
