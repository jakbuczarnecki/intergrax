# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
import time
import pytest

from intergrax.queueing.providers.rabbitmq.rabbitmq_message_producer import (
    RabbitMQMessageProducer,
)
from intergrax.queueing.providers.rabbitmq.rabbitmq_message_consumer import (
    RabbitMQMessageConsumer,
)

pytestmark = pytest.mark.integration


def test_rabbitmq_transport_roundtrip() -> None:
    queue_name = f"intergrax-transport-{uuid.uuid4()}"

    producer = RabbitMQMessageProducer(
        host="localhost",
        username="intergrax",
        password="intergrax",
    )

    consumer = RabbitMQMessageConsumer(
        host="localhost",
        queue=queue_name,
        username="intergrax",
        password="intergrax",
    )

    payload = b"transport-test-payload"

    producer.publish(
        topic=queue_name,
        payload=payload,
    )

    # small wait for broker delivery
    time.sleep(0.2)

    received = consumer.poll(timeout_seconds=1.0)

    assert received is not None
    assert received == payload