# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Kafka integration provider (Phase M.4)."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations._shared.conformance import assert_message_bus
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.kafka.bundle import (
    KafkaIntegrationBundle,
    create_kafka_integration,
    create_kafka_message_bus,
)
from intergrax.integrations.providers.kafka.register import register_kafka_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug
from intergrax.queueing.providers.kafka.kafka_task_queue import KafkaTaskQueue

pytestmark = pytest.mark.unit


class InMemoryKVStore(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str) -> Optional[bytes]:
        return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: Optional[bytes],
        new_value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        current = self.get(tenant_id, key)
        if expected is None and current is not None:
            return False
        if expected is not None and current != expected:
            return False
        self.set(tenant_id, key, new_value, ttl_seconds=ttl_seconds)
        return True


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


@pytest.fixture
def kv_store() -> InMemoryKVStore:
    return InMemoryKVStore()


@pytest.fixture
def mock_producer() -> MagicMock:
    return MagicMock()


def test_create_kafka_integration_bundle(kv_store: InMemoryKVStore, mock_producer: MagicMock) -> None:
    bundle = create_kafka_integration(
        kv_store=kv_store,
        topic="lab-tasks",
        producer=mock_producer,
        consumer=MagicMock(),
    )

    assert isinstance(bundle, KafkaIntegrationBundle)
    assert isinstance(bundle.message_bus, KafkaTaskQueue)
    assert bundle.config.topic == "lab-tasks"
    assert bundle.kv_store is kv_store


def test_create_kafka_message_bus_requires_kv_store(mock_producer: MagicMock) -> None:
    with pytest.raises(ValueError, match="kv_store"):
        create_kafka_message_bus(producer=mock_producer)


def test_register_and_resolve_via_profile(
    kv_store: InMemoryKVStore,
    mock_producer: MagicMock,
) -> None:
    register_kafka_integration()
    profile = IntegrationProfile(message_bus=IntegrationSlug.KAFKA)

    bus = resolve(
        IntegrationCategory.MESSAGE_BUS,
        profile=profile,
        config={"kv_store": kv_store, "producer": mock_producer, "topic": "test-topic"},
    )

    assert_message_bus(bus)
    assert isinstance(bus, KafkaTaskQueue)


def test_register_default_integrations_includes_kafka(
    kv_store: InMemoryKVStore,
    mock_producer: MagicMock,
) -> None:
    register_default_integrations()
    profile = IntegrationProfile(message_bus=IntegrationSlug.KAFKA)

    bus = resolve(
        IntegrationCategory.MESSAGE_BUS,
        profile=profile,
        config={"kv_store": kv_store, "producer": mock_producer},
    )

    assert isinstance(bus, KafkaTaskQueue)
