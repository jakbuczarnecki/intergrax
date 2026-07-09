# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import base64
import json
import uuid

from intergrax.background_tasks.events import TaskEvent, TaskEventName
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.providers.message_bus.kafka.config import KafkaIntegrationConfig
from intergrax.integrations.providers.message_bus.kafka.lifecycle import KafkaTaskLifecycleEmitter
from intergrax.queueing.contracts.message_producer import MessageProducer
from intergrax.queueing.contracts.task_queue import (
    TaskHandle,
    TaskRequest,
    TaskStatus,
)
from intergrax.queueing.providers.broker_task_queue_base import BrokerBackedTaskQueueBase


class KafkaTaskQueue(BrokerBackedTaskQueueBase):
    """
    Kafka-backed TaskQueue implementation.

    Composition root: ``intergrax.integrations.providers.message_bus.kafka.create_kafka_integration``.
    """

    def __init__(
        self,
        *,
        producer: MessageProducer,
        config: KafkaIntegrationConfig,
        kv_store: DistributedKVStore,
        lifecycle_emitter: KafkaTaskLifecycleEmitter | None = None,
    ) -> None:
        super().__init__(
            kv_store=kv_store,
            provider_name="kafka",
        )
        self._producer: MessageProducer = producer
        self._config = config
        self._lifecycle_emitter = lifecycle_emitter

    def _emit_enqueue_event(
        self,
        name: TaskEventName,
        *,
        request: TaskRequest,
        task_id: str,
        status: str | None = None,
    ) -> None:
        if self._lifecycle_emitter is None:
            return
        correlation_id = self._correlation_id_from_payload(request.payload)
        self._lifecycle_emitter.emit(
            TaskEvent(
                name=name,
                task_id=task_id,
                tenant_id=request.tenant_id,
                run_id=request.run_id,
                task_name=request.task_name,
                provider=self._provider_name,
                correlation_id=correlation_id,
                idempotency_key=request.idempotency_key,
                status=status,
            )
        )

    @staticmethod
    def _correlation_id_from_payload(payload: bytes) -> str | None:
        try:
            raw = json.loads(payload.decode("utf-8"))
        except Exception:
            return None
        if isinstance(raw, dict) and raw.get("correlation_id"):
            return str(raw["correlation_id"])
        return None

    def enqueue(
        self,
        request: TaskRequest,
    ) -> TaskHandle:
        task_id = request.run_id.strip() or str(uuid.uuid4())
        self._emit_enqueue_event(
            TaskEventName.ENQUEUE_REQUESTED,
            request=request,
            task_id=task_id,
            status=TaskStatus.PENDING.value,
        )

        self._kv_store.set(
            tenant_id=request.tenant_id,
            key=self._status_key(task_id),
            value=TaskStatus.PENDING.value.encode("utf-8"),
        )
        self.register_task_index(
            tenant_id=request.tenant_id,
            task_id=task_id,
            task_name=request.task_name,
            status=TaskStatus.PENDING,
        )

        encoded_payload = base64.b64encode(request.payload).decode("ascii")
        message = {
            "task_id": task_id,
            "tenant_id": request.tenant_id,
            "run_id": request.run_id,
            "task_name": request.task_name,
            "provider": self._provider_name,
            "payload": encoded_payload,
            "payload_base64": encoded_payload,
            "idempotency_key": request.idempotency_key,
            "priority": request.priority.value,
            "correlation_id": self._correlation_id_from_payload(request.payload),
        }
        payload_bytes = json.dumps(message, separators=(",", ":"), sort_keys=True).encode("utf-8")

        self._producer.publish(
            topic=self._config.topic,
            payload=payload_bytes,
        )
        self._emit_enqueue_event(
            TaskEventName.ENQUEUED,
            request=request,
            task_id=task_id,
            status=TaskStatus.PENDING.value,
        )

        return TaskHandle(
            task_id=task_id,
            provider="kafka",
            tenant_id=request.tenant_id,
        )
