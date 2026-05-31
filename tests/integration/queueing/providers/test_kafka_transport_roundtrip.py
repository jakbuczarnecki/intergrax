# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
import time
from typing import Optional

import pytest

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_integration

pytestmark = pytest.mark.integration


class _MinimalKVStore(DistributedKVStore):
    def get(self, tenant_id: str, key: str) -> Optional[bytes]:
        return None

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        return None

    def delete(self, tenant_id: str, key: str) -> None:
        return None

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: Optional[bytes],
        new_value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        return True


def test_kafka_transport_roundtrip() -> None:
    broker = "localhost:9092"
    topic = f"intergrax-transport-{uuid.uuid4()}"
    group_id = f"intergrax-transport-group-{uuid.uuid4()}"

    bundle = create_kafka_integration(
        kv_store=_MinimalKVStore(),
        bootstrap_servers=broker,
        topic=topic,
        consumer_group=group_id,
    )

    payload = b"transport-test-payload"

    bundle.producer.publish(
        topic=topic,
        payload=payload,
    )

    time.sleep(0.5)

    received = bundle.consumer.poll(timeout_seconds=1.0)

    assert received is not None
    assert received == payload
