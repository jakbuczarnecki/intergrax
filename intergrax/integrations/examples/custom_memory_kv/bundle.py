# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.examples.custom_memory_kv.adapter import InProcessKeyValueCache
from intergrax.integrations.examples.custom_memory_kv.integration import (
    CUSTOM_MEMORY_KV_PROVIDER_ID,
    CustomMemoryKvIntegration,
    CustomMemoryKvIntegrationConfig,
)

__all__ = ["create_custom_memory_kv_integration"]


def create_custom_memory_kv_integration(
    *,
    client: KeyValueCache | None = None,
    enabled: bool = False,
) -> CustomMemoryKvIntegration:
    if client is not None:
        return CustomMemoryKvIntegration.from_client(client, enabled=enabled)
    return CustomMemoryKvIntegration.from_client(
        InProcessKeyValueCache(),
        enabled=enabled,
    )


def create_custom_memory_kv(**kwargs: object) -> CustomMemoryKvIntegration:
    _ = kwargs
    return create_custom_memory_kv_integration()
