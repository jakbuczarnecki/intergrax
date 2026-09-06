# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register llama_cpp embedding provider in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.llama_cpp.bundle import create_llama_cpp_embedding_provider
from intergrax.integrations.providers.embedding_provider.llama_cpp.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.embedding_provider.llama_cpp.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_llama_cpp_embedding_provider_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_llama_cpp_embedding_provider,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
