# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for vLLM embedding provider."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.vllm.bundle import (
    create_vllm_embedding_provider_integration,
)
from intergrax.integrations.providers.embedding_provider.vllm.integration import (
    VLLM_EMBEDDING_PROVIDER_ID,
    VllmEmbeddingProviderIntegration,
    VllmEmbeddingProviderIntegrationConfig,
)
from intergrax.integrations.providers.embedding_provider.vllm.runtime_binding import (
    VLLM_EMBEDDING_PROVIDER_RUNTIME_BINDER,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="embedding_provider",
    provider_id=VLLM_EMBEDDING_PROVIDER_ID,
    integration_class=VllmEmbeddingProviderIntegration,
    contract_class=EmbeddingProviderIntegrationContract,
    contract_factory=create_vllm_embedding_provider_integration,
    display_name="vLLM",
    config_class=VllmEmbeddingProviderIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=False,
    embedding_runtime_binder=VLLM_EMBEDDING_PROVIDER_RUNTIME_BINDER,
    metadata={
        "source": "explicit_provider_declaration",
        "optional_dependency": "openai",
        "runtime_binding_status": "b3_canonical",
    },
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
