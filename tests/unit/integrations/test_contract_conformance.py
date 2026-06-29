# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import pytest

from intergrax.integrations._shared.conformance import (
    assert_cloud_platform,
    assert_interaction_surface,
    assert_key_value_cache,
    assert_message_bus,
    assert_notification_channel,
    assert_object_storage,
    assert_relational_store,
    assert_search_provider,
)
from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.contracts.interaction_surface import InteractionAdapter, InteractionSurface
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.contracts.message_bus import MessageBus, TaskHandle, TaskRequest, TaskResult, TaskStatus
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.runtime.interactions.models import InboundInteraction
from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.websearch.schemas.search_hit import SearchHit

pytestmark = pytest.mark.unit


class _FakeRelationalStore:
    def connect(self) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        return None

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        return [{"sql": sql, "n": len(params)}]

    def close(self) -> None:
        return None


class _FakeKeyValueCache:
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

    def set_if_absent(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        composite = (tenant_id, key)
        if composite in self._data:
            return False
        self._data[composite] = value
        return True


class _FakeMessageBus(MessageBus):
    def enqueue(self, request: TaskRequest) -> TaskHandle:
        return TaskHandle(task_id="t1", provider="fake", tenant_id=request.tenant_id)

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        return TaskStatus.SUCCEEDED

    def get_result(self, handle: TaskHandle) -> Optional[TaskResult]:
        return TaskResult(status=TaskStatus.SUCCEEDED, output=b"ok")


class _FakeSearchProvider:
    def search(self, query: str, *, limit: int = 10) -> Sequence[SearchHit]:
        return [
            SearchHit(
                provider="fake",
                query_issued=query,
                rank=1,
                title="Example",
                url="https://example.com",
            )
        ][:limit]


class _FakeNotificationChannel:
    async def notify(self, message: NotificationMessage) -> None:
        return None

    def health(self) -> bool:
        return True


class _FakeInteractionSurface(InteractionAdapter):
    @property
    def channel(self) -> str:
        return "fake"

    def can_handle(self, payload: Mapping[str, Any]) -> bool:
        return bool(payload)

    def to_inbound(
        self,
        payload: Mapping[str, Any],
        *,
        tenant_id: str,
        user_id: str,
    ) -> InboundInteraction:
        return InboundInteraction(
            channel=self.channel,
            tenant_id=tenant_id,
            user_id=user_id,
            message=str(payload.get("message", "")),
        )


class _FakeObjectStorage:
    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        return None

    def get(self, key: str):
        return None

    def delete(self, key: str) -> None:
        return None

    def presigned_url(
        self,
        key: str,
        *,
        expires_in_seconds: int = 3600,
        method: str = "GET",
    ) -> str:
        return f"https://example.com/{key}?expires={expires_in_seconds}"

    def close(self) -> None:
        return None


class _FakeCloudPlatform:
    @property
    def slug(self) -> str:
        return "aws"

    @property
    def default_region(self) -> Optional[str]:
        return "eu-central-1"

    def resolve(self, category: str) -> Optional[str]:
        return "s3" if category == "object_storage" else None

    def health(self) -> HealthStatus:
        return HealthStatus(slug=self.slug, healthy=True)


def test_relational_store_conformance() -> None:
    assert_relational_store(_FakeRelationalStore())


def test_key_value_cache_conformance() -> None:
    cache = assert_key_value_cache(_FakeKeyValueCache())
    assert cache.set_if_absent("t1", "k", b"v") is True
    assert cache.set_if_absent("t1", "k", b"v2") is False


def test_message_bus_conformance() -> None:
    bus = assert_message_bus(_FakeMessageBus())
    handle = bus.enqueue(
        TaskRequest(tenant_id="t1", run_id="r1", task_name="demo", payload=b"{}")
    )
    assert bus.get_status(handle) == TaskStatus.SUCCEEDED


def test_search_provider_conformance() -> None:
    provider = assert_search_provider(_FakeSearchProvider())
    hits = provider.search("intergrax")
    assert hits[0].url == "https://example.com"


def test_notification_channel_conformance() -> None:
    assert_notification_channel(_FakeNotificationChannel())


def test_interaction_surface_conformance() -> None:
    surface = assert_interaction_surface(_FakeInteractionSurface())
    inbound = surface.to_inbound({"message": "hi"}, tenant_id="t1", user_id="u1")
    assert inbound.message == "hi"


def test_cloud_platform_conformance() -> None:
    cloud = assert_cloud_platform(_FakeCloudPlatform())
    assert cloud.resolve("object_storage") == "s3"


def test_object_storage_conformance() -> None:
    storage = assert_object_storage(_FakeObjectStorage())
    assert storage.presigned_url("artifacts/run.zip") == "https://example.com/artifacts/run.zip?expires=3600"


def test_runtime_notification_adapter_requires_health_for_notification_channel() -> None:
    from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
    from intergrax.runtime.notifications.adapters.logging_adapter import LoggingNotificationAdapter

    adapter = LoggingNotificationAdapter()
    assert isinstance(adapter, NotificationAdapter)
    assert not isinstance(adapter, NotificationChannel)


def test_runtime_interaction_adapter_is_interaction_surface() -> None:
    from intergrax.runtime.interactions.adapters.lab_json_adapter import LabJsonInteractionAdapter

    assert_interaction_surface(LabJsonInteractionAdapter())


def test_queueing_task_queue_is_message_bus() -> None:
    from intergrax.queueing.providers.celery.celery_task_queue import CeleryTaskQueue
    from unittest.mock import MagicMock

    bus = CeleryTaskQueue(app=MagicMock())
    assert_message_bus(bus)
