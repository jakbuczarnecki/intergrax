# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register serpapi in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.serpapi.bundle import create_serpapi_search_provider
from intergrax.integrations.providers.search_provider.serpapi.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_serpapi_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_serpapi_search_provider, override=override)
