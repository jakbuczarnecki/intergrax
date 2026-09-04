# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Replicate ml inference host."""

from __future__ import annotations

from intergrax.integrations.providers.ml_inference_host.replicate.bundle import (
    create_replicate_ml_inference_host_integration,
)
from intergrax.integrations.providers.ml_inference_host.replicate.integration import (
    REPLICATE_ML_INFERENCE_HOST_PROVIDER_ID,
    ReplicateMlInferenceHostIntegration,
    ReplicateMlInferenceHostIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.ai import MlInferenceHostIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="ml_inference_host",
    provider_id=REPLICATE_ML_INFERENCE_HOST_PROVIDER_ID,
    integration_class=ReplicateMlInferenceHostIntegration,
    contract_class=MlInferenceHostIntegrationContract,
    contract_factory=create_replicate_ml_inference_host_integration,
    display_name="Replicate",
    config_class=ReplicateMlInferenceHostIntegrationConfig,
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
