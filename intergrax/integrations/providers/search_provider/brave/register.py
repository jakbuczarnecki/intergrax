# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register brave in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.brave.bundle import create_brave_search_provider
from intergrax.integrations.providers.search_provider.brave.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_brave_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_brave_search_provider, override=override)
