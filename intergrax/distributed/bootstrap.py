# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Distributed layer composition root.

Redis: use ``intergrax.integrations.providers.key_value_cache.redis`` (``create_redis_integration``,
``register_redis_integration``). This bootstrap only registers the KV store **class**
for legacy ``DistributedProviderRegistry`` resolution.
"""

from intergrax.distributed.registry import DistributedProviderRegistry
from intergrax.integrations.providers.key_value_cache.redis import RedisKVStore, register_redis_integration


def bootstrap_default_providers(
    registry: DistributedProviderRegistry,
) -> None:
    """
    Register default distributed providers.

    This function must be called during application composition phase.
    """
    register_redis_integration(override=True)
    registry.register("redis", RedisKVStore)
