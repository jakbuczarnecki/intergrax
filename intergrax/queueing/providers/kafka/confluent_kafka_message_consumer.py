# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Optional

from confluent_kafka import Consumer, KafkaException, Message

from intergrax.queueing.contracts.message_consumer import MessageConsumer


class ConfluentKafkaMessageConsumer(MessageConsumer):
    """
    Production-grade Kafka MessageConsumer implementation
    based on confluent-kafka (librdkafka).

    This adapter:
    - isolates vendor API
    - returns raw bytes only
    - does not expose confluent types outside
    """

    def __init__(
        self,
        *,
        bootstrap_servers: str,
        group_id: str,
        topic: str,
        extra_config: Optional[Dict[str, str]] = None,
    ) -> None:
        config: Dict[str, str] = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": "false",
        }

        if extra_config:
            config.update(extra_config)

        self._consumer: Consumer = Consumer(config)
        self._topic: str = topic
        self._last_message: Message | None = None

        self._consumer.subscribe([self._topic])

    def poll(self, *, timeout_seconds: float) -> Optional[bytes]:
        """
        Retrieve next message payload from Kafka.

        Args:
            timeout_seconds: Maximum time to wait for a message.

        Returns:
            - bytes payload if message available
            - None if no message available
        """

        msg = self._consumer.poll(timeout=timeout_seconds)

        if msg is None:
            return None

        if msg.error():
            raise KafkaException(msg.error())

        self._last_message = msg
        value: Optional[bytes] = msg.value()

        return value

    def commit(self) -> None:
        """Commit the last polled message offset after terminal task handling."""
        if self._last_message is None:
            return
        self._consumer.commit(message=self._last_message, asynchronous=False)
        self._last_message = None
