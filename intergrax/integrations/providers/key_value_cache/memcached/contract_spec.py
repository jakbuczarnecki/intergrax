# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Memcached key value cache."""

from __future__ import annotations

from intergrax.integrations.providers.key_value_cache.memcached.bundle import (
    create_memcached_key_value_cache_integration,
)
from intergrax.integrations.providers.key_value_cache.memcached.integration import (
    MEMCACHED_KEY_VALUE_CACHE_PROVIDER_ID,
    MemcachedKeyValueCacheIntegration,
    MemcachedKeyValueCacheIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import KeyValueCacheIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="key_value_cache",
    provider_id=MEMCACHED_KEY_VALUE_CACHE_PROVIDER_ID,
    integration_class=MemcachedKeyValueCacheIntegration,
    contract_class=KeyValueCacheIntegrationContract,
    contract_factory=create_memcached_key_value_cache_integration,
    display_name="Memcached",
    config_class=MemcachedKeyValueCacheIntegrationConfig,
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
