# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.distributed.registry import DistributedProviderRegistry
from intergrax.distributed.bootstrap import bootstrap_default_providers
from intergrax.distributed.providers.redis_kv_store import RedisKVStore

pytestmark = pytest.mark.unit

def test_bootstrap_registers_redis_provider() -> None:
    registry = DistributedProviderRegistry()

    bootstrap_default_providers(registry)

    provider_cls = registry.get_provider("redis")

    assert provider_cls is RedisKVStore