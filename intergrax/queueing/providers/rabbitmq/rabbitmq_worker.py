# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.queueing.contracts.message_consumer import MessageConsumer
from intergrax.queueing.providers.broker_worker_base import BrokerWorkerBase
from intergrax.runtime.background_execution.identity_persistence import (
    BackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)


class RabbitMQWorker(BrokerWorkerBase):
    """
    Thin RabbitMQ worker adapter.

    Responsibilities:
    - Poll transport via MessageConsumer
    - Delegate raw payload to BrokerWorkerBase.process_message()

    Does NOT:
    - Implement execution logic
    - Implement retry loop
    - Implement DLQ
    - Expose vendor API
    """

    def __init__(
        self,
        *,
        consumer: MessageConsumer,
        registry,
        kv_store,
        idempotency_store=None,
        identity_persistence: BackgroundExecutionIdentityPersistence,
        causal_evidence_persistence: CausalEvidencePersistence,
        poll_timeout_seconds: float = 1.0,
    ) -> None:
        super().__init__(
            registry=registry,
            kv_store=kv_store,
            idempotency_store=idempotency_store,
            identity_persistence=identity_persistence,
            causal_evidence_persistence=causal_evidence_persistence,
        )
        self._consumer: MessageConsumer = consumer
        self._poll_timeout_seconds = poll_timeout_seconds

    def start(self) -> None:
        """
        Start infinite polling loop.

        This is a blocking call.
        """

        while True:
            payload = self._consumer.poll(
                timeout_seconds=self._poll_timeout_seconds
            )

            if payload is None:
                continue

            self.process_message(raw_payload=payload)