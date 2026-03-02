# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict

from confluent_kafka import Producer

from intergrax.queueing.contracts.message_producer import MessageProducer


class ConfluentKafkaMessageProducer(MessageProducer):
    """
    Production-grade Kafka MessageProducer implementation
    based on confluent-kafka (librdkafka).

    This adapter:
    - isolates vendor API
    - ensures deterministic publish (flush after produce)
    - does not expose confluent types outside
    """

    def __init__(
        self,
        *,
        bootstrap_servers: str,
        extra_config: Dict[str, str] | None = None,
    ) -> None:
        config: Dict[str, str] = {
            "bootstrap.servers": bootstrap_servers,
            "acks": "all",
            "retries": "3",
        }

        if extra_config:
            config.update(extra_config)

        self._producer: Producer = Producer(config)

    def publish(
        self,
        topic: str,
        payload: bytes,
    ) -> None:
        """
        Publish message synchronously.

        We:
        - produce
        - poll(0) to serve delivery callbacks
        - flush to ensure delivery

        This guarantees deterministic behavior for task queue.
        """

        self._producer.produce(
            topic=topic,
            value=payload,
        )

        # Serve delivery callbacks (non-blocking)
        self._producer.poll(0)

        # Ensure message is delivered
        remaining: int = self._producer.flush(timeout=10.0)

        if remaining != 0:
            raise RuntimeError(
                f"Kafka producer failed to deliver {remaining} message(s)"
            )