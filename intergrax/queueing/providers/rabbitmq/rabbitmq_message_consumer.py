# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

import pika
from pika.adapters.blocking_connection import BlockingChannel

from intergrax.queueing.contracts.message_consumer import MessageConsumer


class RabbitMQMessageConsumer(MessageConsumer):
    """
    Production-grade RabbitMQ MessageConsumer implementation
    based on pika.

    This adapter:
    - isolates vendor API
    - returns raw bytes only
    - does not expose pika types outside
    """

    def __init__(
        self,
        *,
        host: str,
        port: int = 5672,
        virtual_host: str = "/",
        queue: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        credentials = None
        if username is not None and password is not None:
            credentials = pika.PlainCredentials(username, password)

        if credentials is None:
            parameters = pika.ConnectionParameters(
                host=host,
                port=port,
                virtual_host=virtual_host,
            )
        else:
            parameters = pika.ConnectionParameters(
                host=host,
                port=port,
                virtual_host=virtual_host,
                credentials=credentials,
            )

        self._connection: pika.BlockingConnection = pika.BlockingConnection(parameters)
        self._channel: BlockingChannel = self._connection.channel()

        self._queue: str = queue

        # Ensure queue exists (idempotent declaration)
        self._channel.queue_declare(queue=self._queue, durable=True)

    def poll(self, *, timeout_seconds: float) -> Optional[bytes]:
        """
        Poll RabbitMQ queue for a single message.

        Uses basic_get (non-streaming pull model).
        """

        method_frame, properties, body = self._channel.basic_get(
            queue=self._queue,
            auto_ack=True,
        )

        if method_frame is None:
            return None

        return body