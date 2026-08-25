# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from fastapi.testclient import TestClient

from dispute_sim_application.host.factory import create_dispute_sim_backend_app
from dispute_sim_application.host.settings import DisputeSimBackendSettings
from dispute_sim_application.tests.dispute_sim_ac3_projection import (
    build_dispute_sim_test_registry_projection,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.fastapi_core.config import ApiEnvironment
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from tests.unit.queueing.worker.dispatcher_test_kv import DispatcherTestKVStore

pytestmark = pytest.mark.unit


@dataclass
class _QueueWorkerKeyValueCache:
    kv_store: DistributedKVStore

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self.kv_store.get(tenant_id, key)

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        self.kv_store.set(tenant_id, key, value, ttl_seconds=ttl_seconds)

    def delete(self, tenant_id: str, key: str) -> None:
        self.kv_store.delete(tenant_id, key)

    def set_if_absent(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        return self.kv_store.compare_and_set(
            tenant_id,
            key,
            None,
            value,
            ttl_seconds=ttl_seconds,
        )


@pytest.fixture
def dev_settings() -> DisputeSimBackendSettings:
    return DisputeSimBackendSettings(
        environment=ApiEnvironment.DEV,
        route_prefix="/v1/dispute_sim",
        default_agent_id="dispute_intake",
        identity_source="body_or_context",
        cors_allow_origins=frozenset(),
        allowed_hosts=frozenset(),
        openapi_enabled_override=True,
        api_keys_map={},
    )


def test_dispute_sim_backend_builds_with_queue_worker_disabled(
    dev_settings: DisputeSimBackendSettings,
) -> None:
    app = create_dispute_sim_backend_app(
        registry_projection=build_dispute_sim_test_registry_projection(),
        settings=dev_settings,
    )
    client = TestClient(app)
    assert client.get("/health").status_code in {200, 204}


def test_dispute_sim_backend_builds_with_queue_worker_and_platform_storage(
    dev_settings: DisputeSimBackendSettings,
) -> None:
    settings = replace(dev_settings, include_queue_worker=True)
    app = create_dispute_sim_backend_app(
        registry_projection=build_dispute_sim_test_registry_projection(),
        settings=settings,
        document_store=InMemoryDocumentStore(),
        key_value_cache=_QueueWorkerKeyValueCache(kv_store=DispatcherTestKVStore()),
    )
    client = TestClient(app)
    assert client.get("/health").status_code in {200, 204}


def test_dispute_sim_backend_queue_worker_fails_closed_without_platform_storage(
    dev_settings: DisputeSimBackendSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(dev_settings, include_queue_worker=True)
    monkeypatch.setattr(
        "dispute_sim_application.host.factory.apply_queue_worker_environment_profile",
        lambda environment: environment,
    )
    with pytest.raises(ValueError, match="key_value_cache"):
        create_dispute_sim_backend_app(
            registry_projection=build_dispute_sim_test_registry_projection(),
            settings=settings,
        )
