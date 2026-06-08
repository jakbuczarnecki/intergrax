# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register perplexity in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.perplexity.bundle import create_perplexity_search_provider
from intergrax.integrations.providers.search_provider.perplexity.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_perplexity_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_perplexity_search_provider, override=override)
