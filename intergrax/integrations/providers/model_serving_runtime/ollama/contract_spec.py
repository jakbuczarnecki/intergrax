# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Ollama model serving runtime."""

from __future__ import annotations

from intergrax.integrations.providers.model_serving_runtime.ollama.bundle import (
    create_ollama_model_serving_runtime_integration,
)
from intergrax.integrations.providers.model_serving_runtime.ollama.integration import (
    OLLAMA_MODEL_SERVING_RUNTIME_PROVIDER_ID,
    OllamaModelServingRuntimeIntegration,
    OllamaModelServingRuntimeIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import ModelServingRuntimeIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="model_serving_runtime",
    provider_id=OLLAMA_MODEL_SERVING_RUNTIME_PROVIDER_ID,
    integration_class=OllamaModelServingRuntimeIntegration,
    contract_class=ModelServingRuntimeIntegrationContract,
    contract_factory=create_ollama_model_serving_runtime_integration,
    display_name="Ollama",
    config_class=OllamaModelServingRuntimeIntegrationConfig,
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
