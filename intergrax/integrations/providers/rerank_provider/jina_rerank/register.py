# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register jina_rerank in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.rerank_provider.jina_rerank.bundle import create_jina_rerank_provider
from intergrax.integrations.providers.rerank_provider.jina_rerank.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_jina_rerank_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_jina_rerank_provider, override=override)
