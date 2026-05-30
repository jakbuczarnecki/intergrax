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

        from intergrax.integrations.providers.kafka.bundle import build_kafka_transport

        return build_kafka_transport(
            kv_store=kv_store,
            execution_registry=execution_registry,
            idempotency_store=idempotency_store,
            topic=queue_name,
            consumer_group=consumer_group or "intergrax-default",
            bootstrap_servers=config.kafka.bootstrap_servers,
        )

    if config.backend == "rabbitmq":
        if config.rabbitmq is None:
            raise ValueError("RabbitMQTransportConfig must be provided")

        from intergrax.integrations.providers.rabbitmq.bundle import build_rabbitmq_transport

        rmq_cfg: RabbitMQTransportConfig = config.rabbitmq

        return build_rabbitmq_transport(
            kv_store=kv_store,
            execution_registry=execution_registry,
            idempotency_store=idempotency_store,
            queue=queue_name,
            host=rmq_cfg.host,
            port=rmq_cfg.port,
            virtual_host=rmq_cfg.virtual_host,
            username=rmq_cfg.username,
            password=rmq_cfg.password,
        )

    raise ValueError(f"Unsupported transport backend: {config.backend}")