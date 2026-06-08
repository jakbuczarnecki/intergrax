# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register upstash_redis in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.key_value_cache.upstash_redis.bundle import create_upstash_redis_key_value_cache
from intergrax.integrations.providers.key_value_cache.upstash_redis.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_upstash_redis_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_upstash_redis_key_value_cache, override=override)
