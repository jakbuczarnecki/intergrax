# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import base64
import json
import uuid

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.message_producer import MessageProducer
from intergrax.queueing.contracts.task_queue import (
    TaskRequest,
    TaskHandle,
)
from intergrax.queueing.providers.broker_task_queue_base import (
    BrokerBackedTaskQueueBase,
)


class KafkaTaskQueue(BrokerBackedTaskQueueBase):
    """
    Kafka-backed TaskQueue implementation.

    Composition root: ``intergrax.integrations.providers.message_bus.kafka.create_kafka_integration``.
    """

    def __init__(
        self,
        *,
        producer: MessageProducer,
        topic: str,
        kv_store: DistributedKVStore,
    ) -> None:
        super().__init__(
            kv_store=kv_store,
            provider_name="kafka",
        )
        self._producer: MessageProducer = producer
        self._topic: str = topic

    def enqueue(
        self,
        request: TaskRequest,
    ) -> TaskHandle:
        task_id: str = str(uuid.uuid4())

        # Explicitly initialize task state
        self._kv_store.set(
            tenant_id=request.tenant_id,
            key=self._status_key(task_id),
            value=b"PENDING",
        )

        encoded_payload: str = base64.b64encode(request.payload).decode("ascii")

        message = {
            "task_id": task_id,
            "tenant_id": request.tenant_id,
            "run_id": request.run_id,
            "task_name": request.task_name,
            "payload": encoded_payload,
            "idempotency_key": request.idempotency_key,
        }

        # Serialize to raw bytes for transport layer
        payload_bytes: bytes = json.dumps(message).encode("utf-8")

        self._producer.publish(
            topic=self._topic,
            payload=payload_bytes,
        )

        return TaskHandle(
            task_id=task_id,
            provider="kafka",
            tenant_id=request.tenant_id,
        )