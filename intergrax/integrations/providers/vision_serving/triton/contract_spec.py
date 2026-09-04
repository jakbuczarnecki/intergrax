# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Triton vision serving."""

from __future__ import annotations

from intergrax.integrations.providers.vision_serving.triton.bundle import (
    create_triton_vision_serving_integration,
)
from intergrax.integrations.providers.vision_serving.triton.integration import (
    TRITON_VISION_SERVING_PROVIDER_ID,
    TritonVisionServingIntegration,
    TritonVisionServingIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import VisionServingIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="vision_serving",
    provider_id=TRITON_VISION_SERVING_PROVIDER_ID,
    integration_class=TritonVisionServingIntegration,
    contract_class=VisionServingIntegrationContract,
    contract_factory=create_triton_vision_serving_integration,
    display_name="Triton",
    config_class=TritonVisionServingIntegrationConfig,
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
