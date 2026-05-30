# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
import time
from typing import Optional

import pytest

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.providers.rabbitmq.bundle import create_rabbitmq_integration

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


def test_rabbitmq_transport_roundtrip() -> None:
    queue_name = f"intergrax-transport-{uuid.uuid4()}"

    bundle = create_rabbitmq_integration(
        kv_store=_MinimalKVStore(),
        host="localhost",
        username="intergrax",
        password="intergrax",
        queue=queue_name,
    )

    payload = b"transport-test-payload"

    bundle.producer.publish(
        topic=queue_name,
        payload=payload,
    )

    time.sleep(0.2)

    received = bundle.consumer.poll(timeout_seconds=1.0)

    assert received is not None
    assert received == payload
