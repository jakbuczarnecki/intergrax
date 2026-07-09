# © Artur Czarnecki. All rights reserved.

"""LKW Kafka message-bus wiring for platform background-task proof (LKW.4E)."""

from __future__ import annotations

import os

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.providers.key_value_cache.redis.bundle import create_redis_kv_store
from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_message_bus
from intergrax.integrations.registry.catalog_manifests import KAFKA, REDIS
from intergrax.integrations.registry.profile import IntegrationProfile


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def local_workspace_message_bus_enabled() -> bool:
    return _env_bool("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", default=False)


def create_local_workspace_kafka_message_bus(
    *,
    kv_store: DistributedKVStore | None = None,
) -> MessageBus:
    """Composition root for LKW proof Kafka TaskQueue backed by Redis KV state."""
    store = kv_store or create_redis_kv_store()
    return create_kafka_message_bus(kv_store=store)


def materialize_local_workspace_message_bus_profile(
    profile: IntegrationProfile,
) -> IntegrationProfile:
    """
    When message bus is enabled, resolve Redis KV + Kafka bus instances on the profile.

    Catalog ``profile.resolve(MESSAGE_BUS)`` cannot infer ``kv_store`` automatically;
    LKW proof wiring injects a live bus instance instead.
    """
    if not local_workspace_message_bus_enabled():
        return profile

    kv_store = create_redis_kv_store()
    bus = create_local_workspace_kafka_message_bus(kv_store=kv_store)
    options = dict(profile.options)
    options.setdefault(REDIS.slug, {})
    options.setdefault(KAFKA.slug, {})
    return profile.model_copy(
        update={
            "key_value_cache": profile.key_value_cache or REDIS,
            "message_bus": IntegrationBinding.from_instance(bus),
            "options": options,
        }
    )
