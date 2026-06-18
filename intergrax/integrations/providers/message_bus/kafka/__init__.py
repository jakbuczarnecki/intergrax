# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Kafka integration — single public entry for all Kafka-backed Tier-0 facades.

Implementation classes live under ``intergrax.queueing.providers.kafka``;
compose them only through this package.
"""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.message_bus.kafka.config import (
    DEFAULT_BOOTSTRAP_SERVERS,
    DEFAULT_CONSUMER_GROUP,
    DEFAULT_TOPIC,
    ENV_KAFKA_BOOTSTRAP_SERVERS,
    ENV_KAFKA_CONSUMER_GROUP,
    ENV_KAFKA_TOPIC,
    KafkaIntegrationConfig,
)

__all__ = [
    "DEFAULT_BOOTSTRAP_SERVERS",
    "DEFAULT_CONSUMER_GROUP",
    "DEFAULT_TOPIC",
    "ENV_KAFKA_BOOTSTRAP_SERVERS",
    "ENV_KAFKA_CONSUMER_GROUP",
    "ENV_KAFKA_TOPIC",
    "KafkaIntegrationBundle",
    "KafkaIntegrationConfig",
    "build_kafka_transport",
    "create_kafka_integration",
    "create_kafka_message_bus",
    "create_kafka_worker",
    "register_kafka_integration",
    "resolve_kafka_config",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "KafkaIntegrationBundle",
        "build_kafka_transport",
        "create_kafka_integration",
        "create_kafka_message_bus",
        "create_kafka_worker",
        "resolve_kafka_config",
    }
)


def __getattr__(name: str):
    if name == "register_kafka_integration":
        from intergrax.integrations.providers.message_bus.kafka.register import register_kafka_integration

        return register_kafka_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.message_bus.kafka import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
