# © Artur Czarnecki. All rights reserved.

"""Explicit contract declaration for Presidio."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.presidio.bundle import (
    create_presidio_llm_guardrail_integration,
)
from intergrax.integrations.providers.llm_guardrail.presidio.integration import (
    PRESIDIO_PROVIDER_ID,
    PresidioLlmGuardrailIntegration,
    PresidioLlmGuardrailIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import LlmGuardrailIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="llm_guardrail",
    provider_id=PRESIDIO_PROVIDER_ID,
    integration_class=PresidioLlmGuardrailIntegration,
    contract_class=LlmGuardrailIntegrationContract,
    contract_factory=create_presidio_llm_guardrail_integration,
    display_name="Presidio",
    config_class=PresidioLlmGuardrailIntegrationConfig,
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
