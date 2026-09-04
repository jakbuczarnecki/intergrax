# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register cohere_rerank in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.rerank_provider.cohere_rerank.bundle import create_cohere_rerank_provider
from intergrax.integrations.providers.rerank_provider.cohere_rerank.manifest import MANIFEST
from intergrax.integrations.providers.rerank_provider.cohere_rerank.contract_spec import CONTRACT_SPECS
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_cohere_rerank_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_cohere_rerank_provider, override=override, contract_specs=CONTRACT_SPECS)
