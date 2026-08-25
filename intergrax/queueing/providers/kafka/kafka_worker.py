# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.background_tasks.events import TaskEvent, TaskEventName
from intergrax.integrations.providers.message_bus.kafka.config import KafkaIntegrationConfig
from intergrax.integrations.providers.message_bus.kafka.lifecycle import KafkaTaskLifecycleEmitter
from intergrax.queueing.contracts.message_consumer import MessageConsumer
from intergrax.queueing.providers.broker_worker_base import BrokerWorkerBase
from intergrax.runtime.background_execution.identity_persistence import (
    BackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)


class KafkaWorker(BrokerWorkerBase):
    """
    Thin Kafka worker adapter.

    Responsibilities:
    - Poll transport via MessageConsumer
    - Delegate raw payload to BrokerWorkerBase.process_message()
    - Commit Kafka offset after terminal task handling
    """

    def __init__(
        self,
        *,
        consumer: MessageConsumer,
        registry,
        kv_store,
        config: KafkaIntegrationConfig,
        lifecycle_emitter: KafkaTaskLifecycleEmitter | None = None,
        idempotency_store=None,
        identity_persistence: BackgroundExecutionIdentityPersistence,
        causal_evidence_persistence: CausalEvidencePersistence,
        poll_timeout_seconds: float = 1.0,
    ) -> None:
        super().__init__(
            registry=registry,
            kv_store=kv_store,
            idempotency_store=idempotency_store,
            event_emitter=lifecycle_emitter,
            provider_name="kafka",
            identity_persistence=identity_persistence,
            causal_evidence_persistence=causal_evidence_persistence,
        )
        self._consumer: MessageConsumer = consumer
        self._config = config
        self._poll_timeout_seconds = poll_timeout_seconds

    def _emit_acknowledged(
        self,
        *,
        task_id: str,
        tenant_id: str,
        run_id: str,
        task_name: str,
        idempotency_key: str | None,
        correlation_id: str | None,
    ) -> None:
        if self._event_emitter is None:
            return
        self._event_emitter.emit(
            TaskEvent(
                name=TaskEventName.ACKNOWLEDGED,
                task_id=task_id,
                tenant_id=tenant_id,
                run_id=run_id,
                task_name=task_name,
                provider=self._provider_name,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )

    def start(self) -> None:
        """Start infinite polling loop. This is a blocking call."""

        while True:
            payload = self._consumer.poll(timeout_seconds=self._poll_timeout_seconds)

            if payload is None:
                continue

            message = None
            try:
                import json

                message = json.loads(payload.decode("utf-8"))
            except Exception:
                message = None

            try:
                self.process_message(raw_payload=payload)
            except Exception:
                if hasattr(self._consumer, "commit"):
                    self._consumer.commit()
                continue

            if message is not None and hasattr(self._consumer, "commit"):
                self._emit_acknowledged(
                    task_id=str(message.get("task_id", "")),
                    tenant_id=str(message.get("tenant_id", "")),
                    run_id=str(message.get("run_id", "")),
                    task_name=str(message.get("task_name", "")),
                    idempotency_key=message.get("idempotency_key"),
                    correlation_id=message.get("correlation_id"),
                )
                self._consumer.commit()
