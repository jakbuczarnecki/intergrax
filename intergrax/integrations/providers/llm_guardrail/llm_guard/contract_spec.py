# © Artur Czarnecki. All rights reserved.

"""Explicit contract declaration for LLM Guard."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.llm_guard.bundle import (
    create_llm_guard_llm_guardrail_integration,
)
from intergrax.integrations.providers.llm_guardrail.llm_guard.integration import (
    LLM_GUARD_PROVIDER_ID,
    LlmGuardLlmGuardrailIntegration,
    LlmGuardLlmGuardrailIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import LlmGuardrailIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="llm_guardrail",
    provider_id=LLM_GUARD_PROVIDER_ID,
    integration_class=LlmGuardLlmGuardrailIntegration,
    contract_class=LlmGuardrailIntegrationContract,
    contract_factory=create_llm_guard_llm_guardrail_integration,
    display_name="LLM Guard",
    config_class=LlmGuardLlmGuardrailIntegrationConfig,
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
