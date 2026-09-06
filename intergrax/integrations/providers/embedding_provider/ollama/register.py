# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register ollama embedding provider in the integration catalog."""

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
