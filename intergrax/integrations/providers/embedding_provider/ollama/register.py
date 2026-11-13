# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Contribute the ollama embedding provider to the single ``ollama`` catalog row.

The ``ollama`` slug is owned by ``model_serving_runtime/ollama`` (one catalog row per
slug). This module only augments that row with the embedding category and contract
spec, like the other embedding providers that share a slug with another category.
"""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.embedding_provider.ollama.contract_spec import CONTRACT_SPECS
from intergrax.integrations.registry.catalog import augment_integration_contract_specs


def register_ollama_embedding_provider_integration(*, override: bool = False) -> None:
    del override
    augment_integration_contract_specs(
        "ollama",
        categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
        contract_specs=CONTRACT_SPECS,
    )
