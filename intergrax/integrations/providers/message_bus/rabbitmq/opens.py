# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level RabbitMQ component openers — internal to the rabbitmq integration package.

Only this module may construct RabbitMQ producer/consumer/queue/worker instances.
"""

from __future__ import annotations

from typing import Optional

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.providers.message_bus.rabbitmq.config import RabbitMQIntegrationConfig
from intergrax.queueing.contracts.message_consumer import MessageConsumer
from intergrax.queueing.contracts.message_producer import MessageProducer
from intergrax.queueing.contracts.task_queue import TaskQueue
from intergrax.runtime.background_execution.admission_wiring import (
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)


def open_rabbitmq_producer(
    config: RabbitMQIntegrationConfig,
    *,
    producer: Optional[MessageProducer] = None,
) -> MessageProducer:
    if producer is not None:
        return producer
    from intergrax.queueing.providers.rabbitmq.rabbitmq_message_producer import (
        RabbitMQMessageProducer,
    )

    return RabbitMQMessageProducer(
        host=config.host,
        port=config.port,
        virtual_host=config.virtual_host,
        username=config.username,
        password=config.password,
    )


def open_rabbitmq_consumer(
    config: RabbitMQIntegrationConfig,
    *,
    queue: Optional[str] = None,
    consumer: Optional[MessageConsumer] = None,
) -> MessageConsumer:
    if consumer is not None:
        return consumer
    from intergrax.queueing.providers.rabbitmq.rabbitmq_message_consumer import (
        RabbitMQMessageConsumer,
    )

    return RabbitMQMessageConsumer(
        host=config.host,
        port=config.port,
        virtual_host=config.virtual_host,
        queue=queue or config.queue,
        username=config.username,
        password=config.password,
    )


def open_rabbitmq_task_queue(
    config: RabbitMQIntegrationConfig,
    *,
    kv_store: DistributedKVStore,
    queue: Optional[str] = None,
    producer: Optional[MessageProducer] = None,
) -> TaskQueue:
    from intergrax.queueing.providers.rabbitmq.rabbitmq_task_queue import RabbitMQTaskQueue

    resolved_producer = open_rabbitmq_producer(config, producer=producer)
    return RabbitMQTaskQueue(
        producer=resolved_producer,
        queue=queue or config.queue,
        kv_store=kv_store,
    )


def open_rabbitmq_worker(
    config: RabbitMQIntegrationConfig,
    *,
    kv_store: DistributedKVStore,
    registry: object,
    idempotency_store: Optional[IdempotencyStore] = None,
    queue: Optional[str] = None,
    consumer: Optional[MessageConsumer] = None,
    poll_timeout_seconds: float = 1.0,
    causal_evidence_persistence: CausalEvidencePersistence,
) -> object:
    from intergrax.queueing.providers.rabbitmq.rabbitmq_worker import RabbitMQWorker

    resolved_consumer = open_rabbitmq_consumer(
        config,
        queue=queue,
        consumer=consumer,
    )
    admission = wire_background_execution_admission_dependencies(kv_store=kv_store)
    return RabbitMQWorker(
        consumer=resolved_consumer,
        registry=registry,
        kv_store=kv_store,
        idempotency_store=idempotency_store,
        identity_persistence=admission.identity_persistence,
        causal_evidence_persistence=causal_evidence_persistence,
        attempt_lifecycle=admission.attempt_lifecycle,
        execution_terminal=admission.execution_terminal,
        poll_timeout_seconds=poll_timeout_seconds,
    )
