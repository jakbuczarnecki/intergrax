# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Huggingface Hub object storage."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.huggingface_hub.bundle import (
    create_huggingface_hub_object_storage_integration,
)
from intergrax.integrations.providers.object_storage.huggingface_hub.integration import (
    HUGGINGFACE_HUB_OBJECT_STORAGE_PROVIDER_ID,
    HuggingfaceHubObjectStorageIntegration,
    HuggingfaceHubObjectStorageIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="object_storage",
    provider_id=HUGGINGFACE_HUB_OBJECT_STORAGE_PROVIDER_ID,
    integration_class=HuggingfaceHubObjectStorageIntegration,
    contract_class=ObjectStorageIntegrationContract,
    contract_factory=create_huggingface_hub_object_storage_integration,
    display_name="Huggingface Hub",
    config_class=HuggingfaceHubObjectStorageIntegrationConfig,
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
