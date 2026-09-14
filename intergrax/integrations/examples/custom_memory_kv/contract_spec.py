# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.integrations.examples.custom_memory_kv.bundle import (
    create_custom_memory_kv_integration,
)
from intergrax.integrations.examples.custom_memory_kv.integration import (
    CUSTOM_MEMORY_KV_PROVIDER_ID,
    CustomMemoryKvIntegration,
    CustomMemoryKvIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import KeyValueCacheIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="key_value_cache",
    provider_id=CUSTOM_MEMORY_KV_PROVIDER_ID,
    integration_class=CustomMemoryKvIntegration,
    contract_class=KeyValueCacheIntegrationContract,
    contract_factory=create_custom_memory_kv_integration,
    display_name="Custom memory KV",
    config_class=CustomMemoryKvIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "example_integration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
