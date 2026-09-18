# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.integrations.examples.custom_memory_kv.integration import (
    CustomMemoryKvIntegration,
    CustomMemoryKvIntegrationConfig,
)
from intergrax.integrations.examples.custom_memory_kv.bundle import (
    create_custom_memory_kv_integration,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import KeyValueCacheIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

FIXTURE_EP_KV_PROVIDER_ID = "fixture_ep_kv"

CONTRACT_SPEC = declare_integration_contract(
    category="key_value_cache",
    provider_id=FIXTURE_EP_KV_PROVIDER_ID,
    integration_class=CustomMemoryKvIntegration,
    contract_class=KeyValueCacheIntegrationContract,
    contract_factory=create_custom_memory_kv_integration,
    display_name="Fixture entry-point KV",
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
    metadata={"source": "catalog_fixture"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)
