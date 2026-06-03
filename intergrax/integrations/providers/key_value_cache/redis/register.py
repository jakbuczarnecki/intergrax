# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register redis in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.key_value_cache.redis.bundle import create_redis_key_value_cache
from intergrax.integrations.providers.key_value_cache.redis.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_redis_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_redis_key_value_cache, override=override)
