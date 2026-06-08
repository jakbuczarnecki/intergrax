# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register semantic_scholar in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.semantic_scholar.bundle import create_semantic_scholar_search_provider
from intergrax.integrations.providers.search_provider.semantic_scholar.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_semantic_scholar_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_semantic_scholar_search_provider, override=override)
