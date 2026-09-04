# © Artur Czarnecki. All rights reserved.

"""Explicit contract declaration for Azure Content Safety."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.azure_content_safety.bundle import (
    create_azure_content_safety_llm_guardrail_integration,
)
from intergrax.integrations.providers.llm_guardrail.azure_content_safety.integration import (
    AZURE_CONTENT_SAFETY_PROVIDER_ID,
    AzureContentSafetyLlmGuardrailIntegration,
    AzureContentSafetyLlmGuardrailIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import LlmGuardrailIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="llm_guardrail",
    provider_id=AZURE_CONTENT_SAFETY_PROVIDER_ID,
    integration_class=AzureContentSafetyLlmGuardrailIntegration,
    contract_class=LlmGuardrailIntegrationContract,
    contract_factory=create_azure_content_safety_llm_guardrail_integration,
    display_name="Azure Content Safety",
    config_class=AzureContentSafetyLlmGuardrailIntegrationConfig,
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
