# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from fastapi.testclient import TestClient

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.fastapi_core.config import ApiEnvironment
from legal_application.host.factory import create_legal_backend_app
from legal_application.host.settings import LegalBackendSettings
from legal_application.tests.legal_ac3_projection import (
    build_legal_host_test_manifest,
    build_legal_test_registry_projection,
)
from testing_support.host_fixture_wiring import (
    install_diagnostic_cursor_secret,
    install_host_llm_stub,
    reference_host_document_store,
)
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
def dev_settings() -> LegalBackendSettings:
    return LegalBackendSettings(
        environment=ApiEnvironment.DEV,
        legal_product_profile="strict_legal",
        legal_llm_provider="ollama",
        legal_default_agent_id="legal",
        route_prefix="/v1/legal",
        identity_source="body_or_context",
        cors_allow_origins=frozenset(),
        allowed_hosts=frozenset(),
        openapi_enabled_override=True,
        session_sqlite_path=None,
        api_keys_map={},
        enable_rag=True,
        enable_websearch=True,
    )


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    install_host_llm_stub(monkeypatch)


@pytest.fixture
def _diagnostic_cursor_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    install_diagnostic_cursor_secret(monkeypatch)


@pytest.fixture
def _stub_queue_worker_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.applications._shared.queue_worker_wiring import QueueWorkerWiring
    from intergrax.runtime.task.nexus_task_execution_adapter import NexusTaskExecutionAdapter

    def _wire_optional_queue_execution(*, task_runner, **kwargs: object) -> QueueWorkerWiring:
        del kwargs
        return QueueWorkerWiring(execution_adapter=NexusTaskExecutionAdapter(task_runner))

    monkeypatch.setattr(
        "legal_application.host.factory.wire_optional_queue_execution",
        _wire_optional_queue_execution,
    )


@pytest.fixture
def legal_host_test_manifest(monkeypatch: pytest.MonkeyPatch, dev_settings: LegalBackendSettings) -> None:
    monkeypatch.setattr(
        "legal_application.host.factory.build_legal_manifest",
        lambda settings=None: build_legal_host_test_manifest(settings or dev_settings),
    )


def test_legal_backend_builds_with_queue_worker_disabled(
    dev_settings: LegalBackendSettings,
    legal_host_test_manifest: None,
    _stub_host_llm: None,
    _diagnostic_cursor_secret: None,
) -> None:
    app = create_legal_backend_app(
        registry_projection=build_legal_test_registry_projection(dev_settings),
        settings=dev_settings,
        document_store=reference_host_document_store(),
    )
    client = TestClient(app)
    assert client.get("/health").status_code in {200, 204}


def test_legal_backend_builds_with_queue_worker_and_platform_storage(
    dev_settings: LegalBackendSettings,
    legal_host_test_manifest: None,
    _stub_host_llm: None,
    _diagnostic_cursor_secret: None,
    _stub_queue_worker_wiring: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "legal_application.host.factory.build_legal_manifest",
        lambda settings=None: build_legal_host_test_manifest(
            settings or dev_settings,
            strict_platform_backing=True,
        ),
    )
    settings = replace(dev_settings, include_queue_worker=True)
    app = create_legal_backend_app(
        registry_projection=build_legal_test_registry_projection(settings),
        settings=settings,
        document_store=reference_host_document_store(),
        key_value_cache=DispatcherTestKVStore(),
    )
    client = TestClient(app)
    assert client.get("/health").status_code in {200, 204}


def test_legal_backend_queue_worker_fails_closed_without_platform_storage(
    dev_settings: LegalBackendSettings,
    legal_host_test_manifest: None,
    _stub_host_llm: None,
    _diagnostic_cursor_secret: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(dev_settings, include_queue_worker=True)
    monkeypatch.setattr(
        "legal_application.host.factory.apply_queue_worker_environment_profile",
        lambda environment: environment,
    )
    with pytest.raises(ValueError, match="key_value_cache"):
        create_legal_backend_app(
            registry_projection=build_legal_test_registry_projection(settings),
            settings=settings,
            document_store=reference_host_document_store(),
        )
