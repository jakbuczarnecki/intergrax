# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register bing in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.bing.bundle import create_bing_search_provider
from intergrax.integrations.providers.search_provider.bing.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_bing_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_bing_search_provider, override=override)
