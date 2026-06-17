# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register redis in the integration catalog."""

from __future__ import annotations

from typing import Any

from intergrax.integrations.providers.key_value_cache.redis.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def _create_redis_key_value_cache(**kwargs: Any) -> Any:
    from intergrax.integrations.providers.key_value_cache.redis.bundle import create_redis_key_value_cache

    return create_redis_key_value_cache(**kwargs)


def register_redis_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, _create_redis_key_value_cache, override=override)
