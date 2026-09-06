# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for HuggingFace embedding provider."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.hf.bundle import (
    create_hf_embedding_provider_integration,
)
from intergrax.integrations.providers.embedding_provider.hf.integration import (
    HF_EMBEDDING_PROVIDER_ID,
    HfEmbeddingProviderIntegration,
    HfEmbeddingProviderIntegrationConfig,
)
from intergrax.integrations.providers.embedding_provider.hf.runtime_binding import (
    HF_EMBEDDING_PROVIDER_RUNTIME_BINDER,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.rag.embedding.contracts.runtime_binding_spec import EmbeddingProviderRuntimeBindingSpec
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="embedding_provider",
    provider_id=HF_EMBEDDING_PROVIDER_ID,
    integration_class=HfEmbeddingProviderIntegration,
    contract_class=EmbeddingProviderIntegrationContract,
    contract_factory=create_hf_embedding_provider_integration,
    display_name="HuggingFace",
    config_class=HfEmbeddingProviderIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=False,
    runtime_binding=EmbeddingProviderRuntimeBindingSpec(
        binder=HF_EMBEDDING_PROVIDER_RUNTIME_BINDER,
    ),
    metadata={
        "source": "explicit_provider_declaration",
        "optional_dependency": "sentence-transformers",
        "optional_extra": "rag-local-embeddings",
        "runtime_binding_status": "b3_canonical",
    },
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
