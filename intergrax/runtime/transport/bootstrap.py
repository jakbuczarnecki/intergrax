# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.runtime.transport.config import (
    TransportConfig,
    KafkaTransportConfig,
    RabbitMQTransportConfig,
)
from intergrax.runtime.transport.bundle import TransportBundle

from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.distributed.contracts.kv_store import DistributedKVStore

from intergrax.queueing.providers.kafka.confluent_kafka_message_producer import (
    ConfluentKafkaMessageProducer,
)
from intergrax.queueing.providers.kafka.confluent_kafka_message_consumer import (
    ConfluentKafkaMessageConsumer,
)
from intergrax.queueing.providers.kafka.kafka_task_queue import KafkaTaskQueue
from intergrax.queueing.providers.kafka.kafka_worker import KafkaWorker

from intergrax.queueing.providers.rabbitmq.rabbitmq_message_producer import (
    RabbitMQMessageProducer,
)
from intergrax.queueing.providers.rabbitmq.rabbitmq_message_consumer import (
    RabbitMQMessageConsumer,
)
from intergrax.queueing.providers.rabbitmq.rabbitmq_task_queue import (
    RabbitMQTaskQueue,
)
from intergrax.queueing.providers.rabbitmq.rabbitmq_worker import (
    RabbitMQWorker,
)


def build_transport(
    *,
    config: TransportConfig,
    execution_registry: TaskExecutionRegistry,
    kv_store: DistributedKVStore,
    idempotency_store: Optional[IdempotencyStore],
    queue_name: str,
    consumer_group: Optional[str] = None,
) -> TransportBundle:
    """
    Build and wire transport components for selected backend.

    This function is the single composition root for queueing transport.
    """

    if config.backend == "kafka":
        if config.kafka is None:
            raise ValueError("KafkaTransportConfig must be provided")

        kafka_cfg: KafkaTransportConfig = config.kafka

        producer = ConfluentKafkaMessageProducer(
            bootstrap_servers=kafka_cfg.bootstrap_servers,
        )

        consumer = ConfluentKafkaMessageConsumer(
            bootstrap_servers=kafka_cfg.bootstrap_servers,
            topic=queue_name,
            group_id=consumer_group or "intergrax-default",
            extra_config={"auto.offset.reset": "earliest"},
        )

        task_queue = KafkaTaskQueue(
            producer=producer,
            topic=queue_name,
            kv_store=kv_store,
        )

        worker = KafkaWorker(
            consumer=consumer,
            registry=execution_registry,
            kv_store=kv_store,
            idempotency_store=idempotency_store,
        )

        return TransportBundle(
            task_queue=task_queue,
            worker=worker,
        )

    if config.backend == "rabbitmq":
        if config.rabbitmq is None:
            raise ValueError("RabbitMQTransportConfig must be provided")

        rmq_cfg: RabbitMQTransportConfig = config.rabbitmq

        producer = RabbitMQMessageProducer(
            host=rmq_cfg.host,
            port=rmq_cfg.port,
            virtual_host=rmq_cfg.virtual_host,
            username=rmq_cfg.username,
            password=rmq_cfg.password,
        )

        consumer = RabbitMQMessageConsumer(
            host=rmq_cfg.host,
            queue=queue_name,
            port=rmq_cfg.port,
            virtual_host=rmq_cfg.virtual_host,
            username=rmq_cfg.username,
            password=rmq_cfg.password,
        )

        task_queue = RabbitMQTaskQueue(
            producer=producer,
            queue=queue_name,
            kv_store=kv_store,
        )

        worker = RabbitMQWorker(
            consumer=consumer,
            registry=execution_registry,
            kv_store=kv_store,
            idempotency_store=idempotency_store,
        )

        return TransportBundle(
            task_queue=task_queue,
            worker=worker,
        )

    raise ValueError(f"Unsupported transport backend: {config.backend}")