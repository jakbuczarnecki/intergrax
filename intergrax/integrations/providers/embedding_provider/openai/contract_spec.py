# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for OpenAI embedding provider."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.openai.bundle import (
    create_openai_embedding_provider_integration,
)
from intergrax.integrations.providers.embedding_provider.openai.integration import (
    OPENAI_EMBEDDING_PROVIDER_ID,
    OpenaiEmbeddingProviderIntegration,
    OpenaiEmbeddingProviderIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="embedding_provider",
    provider_id=OPENAI_EMBEDDING_PROVIDER_ID,
    integration_class=OpenaiEmbeddingProviderIntegration,
    contract_class=EmbeddingProviderIntegrationContract,
    contract_factory=create_openai_embedding_provider_integration,
    display_name="OpenAI",
    config_class=OpenaiEmbeddingProviderIntegrationConfig,
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
