# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for llama.cpp embedding provider."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.llama_cpp.bundle import (
    create_llama_cpp_embedding_provider_integration,
)
from intergrax.integrations.providers.embedding_provider.llama_cpp.integration import (
    LLAMA_CPP_EMBEDDING_PROVIDER_ID,
    LlamaCppEmbeddingProviderIntegration,
    LlamaCppEmbeddingProviderIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="embedding_provider",
    provider_id=LLAMA_CPP_EMBEDDING_PROVIDER_ID,
    integration_class=LlamaCppEmbeddingProviderIntegration,
    contract_class=EmbeddingProviderIntegrationContract,
    contract_factory=create_llama_cpp_embedding_provider_integration,
    display_name="llama.cpp",
    config_class=LlamaCppEmbeddingProviderIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=False,
    supports_health_check=False,
    metadata={
        "source": "explicit_provider_declaration",
        "optional_dependency": "openai",
        "runtime_binding_status": "pending_b3",
    },
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
