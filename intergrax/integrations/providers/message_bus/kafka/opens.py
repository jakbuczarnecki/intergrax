# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Kafka component openers — internal to the kafka integration package.

Only this module may construct Kafka producer/consumer/queue/worker instances.
"""

from __future__ import annotations

from typing import Optional

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.providers.message_bus.kafka.config import KafkaIntegrationConfig
from intergrax.queueing.contracts.message_consumer import MessageConsumer
from intergrax.queueing.contracts.message_producer import MessageProducer
from intergrax.queueing.contracts.task_queue import TaskQueue
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)


def open_kafka_producer(
    config: KafkaIntegrationConfig,
    *,
    producer: Optional[MessageProducer] = None,
) -> MessageProducer:
    if producer is not None:
        return producer
    from intergrax.queueing.providers.kafka.confluent_kafka_message_producer import (
        ConfluentKafkaMessageProducer,
    )

    return ConfluentKafkaMessageProducer(bootstrap_servers=config.bootstrap_servers)


def open_kafka_consumer(
    config: KafkaIntegrationConfig,
    *,
    topic: Optional[str] = None,
    consumer_group: Optional[str] = None,
    consumer: Optional[MessageConsumer] = None,
) -> MessageConsumer:
    if consumer is not None:
        return consumer
    from intergrax.queueing.providers.kafka.confluent_kafka_message_consumer import (
        ConfluentKafkaMessageConsumer,
    )

    return ConfluentKafkaMessageConsumer(
        bootstrap_servers=config.bootstrap_servers,
        topic=topic or config.topic,
        group_id=consumer_group or config.consumer_group,
    )


def _open_kafka_lifecycle_emitter(
    config: KafkaIntegrationConfig,
    *,
    producer: MessageProducer,
) -> "KafkaTaskLifecycleEmitter":
    from intergrax.integrations.providers.message_bus.kafka.lifecycle import KafkaTaskLifecycleEmitter

    return KafkaTaskLifecycleEmitter(
        producer=producer,
        events_topic=config.events_topic,
        status_topic=config.status_topic,
        results_topic=config.results_topic,
    )


def open_kafka_task_queue(
    config: KafkaIntegrationConfig,
    *,
    kv_store: DistributedKVStore,
    topic: Optional[str] = None,
    producer: Optional[MessageProducer] = None,
) -> TaskQueue:
    from intergrax.queueing.providers.kafka.kafka_task_queue import KafkaTaskQueue

    resolved_producer = open_kafka_producer(config, producer=producer)
    resolved_config = config.model_copy(update={"topic": topic or config.topic})
    lifecycle_emitter = _open_kafka_lifecycle_emitter(resolved_config, producer=resolved_producer)
    return KafkaTaskQueue(
        producer=resolved_producer,
        config=resolved_config,
        kv_store=kv_store,
        lifecycle_emitter=lifecycle_emitter,
    )


def open_kafka_worker(
    config: KafkaIntegrationConfig,
    *,
    kv_store: DistributedKVStore,
    registry: object,
    idempotency_store: Optional[IdempotencyStore] = None,
    topic: Optional[str] = None,
    consumer_group: Optional[str] = None,
    consumer: Optional[MessageConsumer] = None,
    poll_timeout_seconds: float = 1.0,
    causal_evidence_persistence: CausalEvidencePersistence,
) -> object:
    from intergrax.queueing.providers.kafka.kafka_worker import KafkaWorker
    from intergrax.runtime.background_execution.identity_persistence import (
        wire_background_execution_identity_persistence,
    )

    resolved_consumer = open_kafka_consumer(
        config,
        topic=topic,
        consumer_group=consumer_group,
        consumer=consumer,
    )
    resolved_producer = open_kafka_producer(config)
    lifecycle_emitter = _open_kafka_lifecycle_emitter(config, producer=resolved_producer)
    return KafkaWorker(
        consumer=resolved_consumer,
        registry=registry,
        kv_store=kv_store,
        config=config,
        lifecycle_emitter=lifecycle_emitter,
        idempotency_store=idempotency_store,
        identity_persistence=wire_background_execution_identity_persistence(
            kv_store=kv_store,
        ),
        causal_evidence_persistence=causal_evidence_persistence,
        poll_timeout_seconds=poll_timeout_seconds,
    )
