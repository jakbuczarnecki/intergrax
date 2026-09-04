# © Artur Czarnecki. All rights reserved.

"""Explicit contract declaration for Bedrock Guardrails."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.bundle import (
    create_bedrock_guardrails_llm_guardrail_integration,
)
from intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.integration import (
    BEDROCK_GUARDRAILS_PROVIDER_ID,
    BedrockGuardrailsLlmGuardrailIntegration,
    BedrockGuardrailsLlmGuardrailIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import LlmGuardrailIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="llm_guardrail",
    provider_id=BEDROCK_GUARDRAILS_PROVIDER_ID,
    integration_class=BedrockGuardrailsLlmGuardrailIntegration,
    contract_class=LlmGuardrailIntegrationContract,
    contract_factory=create_bedrock_guardrails_llm_guardrail_integration,
    display_name="Bedrock Guardrails",
    config_class=BedrockGuardrailsLlmGuardrailIntegrationConfig,
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
