# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X5 / X5A — Kafka transport failure, recovery, and public consumer lifecycle."""

from __future__ import annotations

import os
import time
import uuid

import pytest

from intergrax.integrations.providers.message_bus.kafka.config import (
    DEFAULT_BOOTSTRAP_SERVERS,
)
from intergrax.queueing.providers.kafka.confluent_kafka_message_consumer import (
    ConfluentKafkaMessageConsumer,
)
from intergrax.queueing.providers.kafka.confluent_kafka_message_producer import (
    ConfluentKafkaMessageProducer,
)
from testing_support.cross_process_spine.kafka_probe import (
    ensure_kafka_topic,
    kafka_broker_ready,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.obs_diag_x5,
    pytest.mark.obs_diag_x5a,
]

_BOOTSTRAP = os.environ.get(
    "INTERGRAX_KAFKA_BOOTSTRAP_SERVERS",
    DEFAULT_BOOTSTRAP_SERVERS,
).strip()


def _kafka_available() -> bool:
    try:
        kafka_broker_ready(_BOOTSTRAP, timeout_seconds=8.0)
        return True
    except (TimeoutError, OSError):
        return False


def test_kafka_producer_fails_closed_when_broker_unreachable() -> None:
    producer = ConfluentKafkaMessageProducer(bootstrap_servers="127.0.0.1:1")
    with pytest.raises(RuntimeError, match="failed to deliver"):
        producer.publish(
            topic=f"intergrax-x5-unreachable-{uuid.uuid4()}", payload=b"x5"
        )


@pytest.mark.skipif(not _kafka_available(), reason="real Kafka broker unavailable")
def test_kafka_producer_recovery_after_unreachable_endpoint_then_fresh_provider() -> None:
    dead = ConfluentKafkaMessageProducer(bootstrap_servers="127.0.0.1:1")
    with pytest.raises(RuntimeError, match="failed to deliver"):
        dead.publish(topic="intergrax-x5-dead", payload=b"dead")

    topic = f"intergrax-x5-recovery-{uuid.uuid4()}"
    ensure_kafka_topic(_BOOTSTRAP, topic)
    live = ConfluentKafkaMessageProducer(bootstrap_servers=_BOOTSTRAP)
    payload = b"recovered-payload"
    live.publish(topic=topic, payload=payload)

    group_id = f"intergrax-x5-group-{uuid.uuid4()}"
    consumer = ConfluentKafkaMessageConsumer(
        bootstrap_servers=_BOOTSTRAP,
        group_id=group_id,
        topic=topic,
    )
    received = consumer.poll(timeout_seconds=5.0)
    assert received == payload
    consumer.commit()


@pytest.mark.skipif(not _kafka_available(), reason="real Kafka broker unavailable")
def test_kafka_consumer_restart_redelivers_without_commit() -> None:
    topic = f"intergrax-x5-restart-{uuid.uuid4()}"
    ensure_kafka_topic(_BOOTSTRAP, topic)
    payload = b"at-least-once-payload"
    ConfluentKafkaMessageProducer(bootstrap_servers=_BOOTSTRAP).publish(
        topic=topic,
        payload=payload,
    )
    group_id = f"intergrax-x5-restart-group-{uuid.uuid4()}"

    first = ConfluentKafkaMessageConsumer(
        bootstrap_servers=_BOOTSTRAP,
        group_id=group_id,
        topic=topic,
    )
    assert first.poll(timeout_seconds=5.0) == payload
    first.close()
    time.sleep(2.0)

    second = ConfluentKafkaMessageConsumer(
        bootstrap_servers=_BOOTSTRAP,
        group_id=group_id,
        topic=topic,
    )
    redelivered = second.poll(timeout_seconds=5.0)
    assert redelivered == payload
