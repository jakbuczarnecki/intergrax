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


def open_kafka_task_queue(
    config: KafkaIntegrationConfig,
    *,
    kv_store: DistributedKVStore,
    topic: Optional[str] = None,
    producer: Optional[MessageProducer] = None,
) -> TaskQueue:
    from intergrax.queueing.providers.kafka.kafka_task_queue import KafkaTaskQueue

    resolved_producer = open_kafka_producer(config, producer=producer)
    return KafkaTaskQueue(
        producer=resolved_producer,
        topic=topic or config.topic,
        kv_store=kv_store,
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
) -> object:
    from intergrax.queueing.providers.kafka.kafka_worker import KafkaWorker

    resolved_consumer = open_kafka_consumer(
        config,
        topic=topic,
        consumer_group=consumer_group,
        consumer=consumer,
    )
    return KafkaWorker(
        consumer=resolved_consumer,
        registry=registry,
        kv_store=kv_store,
        idempotency_store=idempotency_store,
        poll_timeout_seconds=poll_timeout_seconds,
    )
