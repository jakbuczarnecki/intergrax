# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
import time
import pytest

from intergrax.queueing.providers.kafka.confluent_kafka_message_producer import (
    ConfluentKafkaMessageProducer,
)
from intergrax.queueing.providers.kafka.confluent_kafka_message_consumer import (
    ConfluentKafkaMessageConsumer,
)

pytestmark = pytest.mark.integration


def test_kafka_transport_roundtrip() -> None:
    broker = "localhost:9092"
    topic = f"intergrax-transport-{uuid.uuid4()}"
    group_id = f"intergrax-transport-group-{uuid.uuid4()}"

    producer = ConfluentKafkaMessageProducer(
        bootstrap_servers=broker,
    )

    consumer = ConfluentKafkaMessageConsumer(
        bootstrap_servers=broker,
        topic=topic,
        group_id=group_id,
        extra_config={
            "auto.offset.reset": "earliest",
        },
    )

    payload = b"transport-test-payload"

    producer.publish(
        topic=topic,
        payload=payload,
    )

    # allow broker delivery
    time.sleep(0.5)

    received = consumer.poll(timeout_seconds=1.0)

    assert received is not None
    assert received == payload