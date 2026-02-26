# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Distributed layer composition root.

This module is responsible for wiring concrete distributed providers
into the DistributedProviderRegistry.

It is the only place where registry and concrete providers are coupled.
"""

from intergrax.distributed.registry import DistributedProviderRegistry
from intergrax.distributed.providers.redis_kv_store import RedisKVStore


def bootstrap_default_providers(
    registry: DistributedProviderRegistry,
) -> None:
    """
    Register default distributed providers.

    This function must be called during application composition phase.
    """
    registry.register("redis", RedisKVStore)