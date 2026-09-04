# © Artur Czarnecki. All rights reserved.

"""Explicit contract declaration for Llama Guard."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.llama_guard.bundle import (
    create_llama_guard_llm_guardrail_integration,
)
from intergrax.integrations.providers.llm_guardrail.llama_guard.integration import (
    LLAMA_GUARD_PROVIDER_ID,
    LlamaGuardLlmGuardrailIntegration,
    LlamaGuardLlmGuardrailIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import LlmGuardrailIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="llm_guardrail",
    provider_id=LLAMA_GUARD_PROVIDER_ID,
    integration_class=LlamaGuardLlmGuardrailIntegration,
    contract_class=LlmGuardrailIntegrationContract,
    contract_factory=create_llama_guard_llm_guardrail_integration,
    display_name="Llama Guard",
    config_class=LlamaGuardLlmGuardrailIntegrationConfig,
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
