# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for OpenAI managed retrieval."""

from __future__ import annotations

from intergrax.integrations.providers.managed_retrieval.openai.bundle import (
    create_openai_managed_retrieval_integration,
)
from intergrax.integrations.providers.managed_retrieval.openai.integration import (
    OPENAI_MANAGED_RETRIEVAL_PROVIDER_ID,
    OpenAIManagedRetrievalIntegration,
    OpenAIManagedRetrievalIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.managed_retrieval import (
    ManagedRetrievalIntegrationContract,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="managed_retrieval",
    provider_id=OPENAI_MANAGED_RETRIEVAL_PROVIDER_ID,
    integration_class=OpenAIManagedRetrievalIntegration,
    contract_class=ManagedRetrievalIntegrationContract,
    contract_factory=create_openai_managed_retrieval_integration,
    display_name="OpenAI",
    config_class=OpenAIManagedRetrievalIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
