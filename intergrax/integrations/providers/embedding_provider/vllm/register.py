# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register vllm embedding provider in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.vllm.bundle import create_vllm_embedding_provider
from intergrax.integrations.providers.embedding_provider.vllm.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.embedding_provider.vllm.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_vllm_embedding_provider_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_vllm_embedding_provider,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
