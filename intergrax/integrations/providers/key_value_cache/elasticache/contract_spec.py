# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Elasticache key value cache."""

from __future__ import annotations

from intergrax.integrations.providers.key_value_cache.elasticache.bundle import (
    create_elasticache_key_value_cache_integration,
)
from intergrax.integrations.providers.key_value_cache.elasticache.integration import (
    ELASTICACHE_KEY_VALUE_CACHE_PROVIDER_ID,
    ElasticacheKeyValueCacheIntegration,
    ElasticacheKeyValueCacheIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import KeyValueCacheIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="key_value_cache",
    provider_id=ELASTICACHE_KEY_VALUE_CACHE_PROVIDER_ID,
    integration_class=ElasticacheKeyValueCacheIntegration,
    contract_class=KeyValueCacheIntegrationContract,
    contract_factory=create_elasticache_key_value_cache_integration,
    display_name="Elasticache",
    config_class=ElasticacheKeyValueCacheIntegrationConfig,
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
