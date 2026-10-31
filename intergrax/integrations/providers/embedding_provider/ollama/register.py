# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register ollama embedding provider in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.ollama.bundle import create_ollama_embedding_provider
from intergrax.integrations.providers.embedding_provider.ollama.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.embedding_provider.ollama.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_ollama_embedding_provider_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_ollama_embedding_provider,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
