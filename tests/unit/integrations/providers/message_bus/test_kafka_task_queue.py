# © Artur Czarnecki. All rights reserved.

"""Kafka TaskQueue contract tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.providers.message_bus.kafka.config import KafkaIntegrationConfig
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.providers.kafka.kafka_task_queue import KafkaTaskQueue
from tests.unit.integrations.providers.message_bus.test_kafka import InMemoryKVStore

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_kafka_task_queue_enqueue_returns_kafka_provider() -> None:
    producer = MagicMock()
    kv = InMemoryKVStore()
    config = KafkaIntegrationConfig(topic="intergrax.tasks")
    queue = KafkaTaskQueue(producer=producer, config=config, kv_store=kv)
    handle = queue.enqueue(
        TaskRequest(
            tenant_id="tenant-a",
            run_id="run-proof-1",
            task_name="lkw.background_ingest.v1",
            payload=b'{"tenant_id":"tenant-a"}',
        )
    )
    assert handle.provider == "kafka"
    assert handle.task_id == "run-proof-1"
    assert queue.get_status(handle) == TaskStatus.PENDING
    producer.publish.assert_called_once()
    assert producer.publish.call_args.kwargs["topic"] == "intergrax.tasks"
    payload = producer.publish.call_args.kwargs["payload"]
    message = json.loads(payload.decode("utf-8"))
    assert message["task_name"] == "lkw.background_ingest.v1"
    assert message["payload_base64"]
