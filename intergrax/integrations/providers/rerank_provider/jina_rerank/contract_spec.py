# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Jina Rerank."""

from __future__ import annotations

from intergrax.integrations.providers.rerank_provider.jina_rerank.bundle import (
    create_jina_rerank_rerank_provider_integration,
)
from intergrax.integrations.providers.rerank_provider.jina_rerank.integration import (
    JINA_RERANK_RERANK_PROVIDER_PROVIDER_ID,
    JinaRerankRerankProviderIntegration,
    JinaRerankRerankProviderIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.search import RerankProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="rerank_provider",
    provider_id=JINA_RERANK_RERANK_PROVIDER_PROVIDER_ID,
    integration_class=JinaRerankRerankProviderIntegration,
    contract_class=RerankProviderIntegrationContract,
    contract_factory=create_jina_rerank_rerank_provider_integration,
    display_name="Jina Rerank",
    config_class=JinaRerankRerankProviderIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
