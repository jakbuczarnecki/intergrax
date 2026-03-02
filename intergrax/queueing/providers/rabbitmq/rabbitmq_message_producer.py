# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

import pika
from pika.adapters.blocking_connection import BlockingChannel
from intergrax.queueing.contracts.message_producer import MessageProducer


class RabbitMQMessageProducer(MessageProducer):
    """
    Production-grade RabbitMQ MessageProducer implementation
    based on pika.

    This adapter:
    - isolates vendor API
    - publishes raw bytes only
    - does not expose pika types outside
    """

    def __init__(
        self,
        *,
        host: str,
        port: int = 5672,
        virtual_host: str = "/",
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

    def publish(
        self,
        topic: str,
        payload: bytes,
    ) -> None:
        """
        Publish raw bytes payload to a given queue.

        Topic is mapped directly to routing_key.
        Default exchange is used.
        """

        self._channel.basic_publish(
            exchange="",
            routing_key=topic,
            body=payload,
        )