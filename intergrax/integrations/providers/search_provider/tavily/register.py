# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register tavily in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.tavily.bundle import create_tavily_search_provider
from intergrax.integrations.providers.search_provider.tavily.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_tavily_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_tavily_search_provider, override=override)
