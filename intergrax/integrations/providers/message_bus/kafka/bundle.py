# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Kafka integration bundle — the single composition root for Kafka in Intergrax.

All runtime wiring (message bus, producer, consumer, worker, transport bundle) MUST use
this module or ``profile.resolve(IntegrationCategory.MESSAGE_BUS)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.providers.message_bus.kafka.config import KafkaIntegrationConfig
from intergrax.integrations.providers.message_bus.kafka.integration import (
    KAFKA_MESSAGE_BUS_PROVIDER_ID,
    KafkaMessageBusClient,
    KafkaMessageBusIntegration,
    KafkaMessageBusIntegrationConfig,
)
from intergrax.integrations.providers.message_bus.kafka.opens import (
    open_kafka_consumer,
    open_kafka_producer,
    open_kafka_task_queue,
    open_kafka_worker,
)
from intergrax.queueing.contracts.message_consumer import MessageConsumer
from intergrax.queueing.contracts.message_producer import MessageProducer
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)


@dataclass(frozen=True)
class KafkaIntegrationBundle:
    """Kafka-backed message bus and transport components sharing one config."""

    config: KafkaIntegrationConfig
    kv_store: DistributedKVStore
    message_bus: MessageBus
    producer: MessageProducer
    consumer: MessageConsumer


def resolve_kafka_config(**overrides: object) -> KafkaIntegrationConfig:
    return KafkaIntegrationConfig.from_env(**overrides)


def _require_kv_store(kv_store: DistributedKVStore | None) -> DistributedKVStore:
    if kv_store is None:
        raise ValueError("Kafka integration requires ``kv_store`` (typically from redis integration).")
    return kv_store


def create_kafka_message_bus_integration(
    *,
    client: KafkaMessageBusClient | None = None,
    enabled: bool = False,
) -> KafkaMessageBusIntegration:
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Kafka message bus integration requires an injected client when enabled=True",
        )
    if client is not None:
        return KafkaMessageBusIntegration.from_client(client, enabled=enabled)
    return KafkaMessageBusIntegration.for_provider(
        provider_id=KAFKA_MESSAGE_BUS_PROVIDER_ID,
        display_name="Kafka",
        config=KafkaMessageBusIntegrationConfig(enabled=enabled),
    )


def create_kafka_integration(
    *,
    kv_store: DistributedKVStore | None = None,
    bootstrap_servers: str | None = None,
    topic: str | None = None,
    consumer_group: str | None = None,
    producer: MessageProducer | None = None,
    consumer: MessageConsumer | None = None,
    **config_overrides: object,
) -> KafkaIntegrationBundle:
    """Single entry point for Kafka — message bus + producer + consumer."""
    overrides: dict[str, object] = dict(config_overrides)
    if bootstrap_servers is not None:
        overrides["bootstrap_servers"] = bootstrap_servers
    if topic is not None:
        overrides["topic"] = topic
    if consumer_group is not None:
        overrides["consumer_group"] = consumer_group

    config = resolve_kafka_config(**overrides)
    store = _require_kv_store(kv_store)
    resolved_producer = open_kafka_producer(config, producer=producer)
    resolved_consumer = open_kafka_consumer(
        config,
        topic=topic,
        consumer_group=consumer_group,
        consumer=consumer,
    )
    bus = open_kafka_task_queue(
        config,
        kv_store=store,
        topic=topic,
        producer=resolved_producer,
    )

    return KafkaIntegrationBundle(
        config=config,
        kv_store=store,
        message_bus=bus,
        producer=resolved_producer,
        consumer=resolved_consumer,
    )


def create_kafka_message_bus(
    *,
    kv_store: DistributedKVStore | None = None,
    bootstrap_servers: str | None = None,
    topic: str | None = None,
    producer: MessageProducer | None = None,
    **config_overrides: object,
) -> KafkaMessageBusIntegration:
    """Catalog factory for ``"kafka"`` / ``MESSAGE_BUS``."""
    if kv_store is None:
        return create_kafka_message_bus_integration(enabled=False)
    bus = create_kafka_integration(
        kv_store=kv_store,
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        producer=producer,
        **config_overrides,
    ).message_bus
    return KafkaMessageBusIntegration.from_client(bus, enabled=True)


def create_kafka_worker(
    *,
    kv_store: DistributedKVStore,
    execution_registry: TaskExecutionRegistry,
    idempotency_store: IdempotencyStore | None = None,
    bootstrap_servers: str | None = None,
    topic: str | None = None,
    consumer_group: str | None = None,
    consumer: MessageConsumer | None = None,
    poll_timeout_seconds: float = 1.0,
    causal_evidence_persistence: CausalEvidencePersistence,
    **config_overrides: object,
) -> object:
    overrides: dict[str, object] = dict(config_overrides)
    if bootstrap_servers is not None:
        overrides["bootstrap_servers"] = bootstrap_servers
    if topic is not None:
        overrides["topic"] = topic
    if consumer_group is not None:
        overrides["consumer_group"] = consumer_group

    config = resolve_kafka_config(**overrides)
    return open_kafka_worker(
        config,
        kv_store=_require_kv_store(kv_store),
        registry=execution_registry,
        idempotency_store=idempotency_store,
        topic=topic,
        consumer_group=consumer_group,
        consumer=consumer,
        poll_timeout_seconds=poll_timeout_seconds,
        causal_evidence_persistence=causal_evidence_persistence,
    )


def build_kafka_transport(
    *,
    kv_store: DistributedKVStore,
    execution_registry: TaskExecutionRegistry,
    idempotency_store: IdempotencyStore | None = None,
    topic: str | None = None,
    consumer_group: str | None = None,
    bootstrap_servers: str | None = None,
    producer: MessageProducer | None = None,
    consumer: MessageConsumer | None = None,
    causal_evidence_persistence: CausalEvidencePersistence,
    **config_overrides: object,
) -> object:
    """
    Build ``TransportBundle`` for Kafka (replaces inline wiring in ``runtime.transport``).
    """
    from intergrax.runtime.transport.bundle import TransportBundle

    bundle = create_kafka_integration(
        kv_store=kv_store,
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        consumer_group=consumer_group,
        producer=producer,
        consumer=consumer,
        **config_overrides,
    )
    worker = open_kafka_worker(
        bundle.config,
        kv_store=bundle.kv_store,
        registry=execution_registry,
        idempotency_store=idempotency_store,
        topic=topic or bundle.config.topic,
        consumer_group=consumer_group or bundle.config.consumer_group,
        consumer=bundle.consumer,
        causal_evidence_persistence=causal_evidence_persistence,
    )
    return TransportBundle(task_queue=bundle.message_bus, worker=worker)
