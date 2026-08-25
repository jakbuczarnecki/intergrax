# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.host_queue_execution_wiring import (
    apply_queue_worker_integration_profile,
    resolve_host_queue_execution_dependencies,
)
from intergrax.distributed.contracts.kv_store import (
    DistributedKVStore,
    DistributedKVStoreProvider,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.registry.profile import IntegrationProfile
from legal_application.host.settings import LegalBackendSettings
from legal_application.host.wiring import build_legal_environment_profile, build_legal_manifest
from legal_application.tests.legal_ac3_projection import build_legal_test_registry_projection
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


def test_apply_queue_worker_integration_profile_adds_platform_bindings() -> None:
    profile = IntegrationProfile.legal_product()
    enriched = apply_queue_worker_integration_profile(profile)
    assert enriched.key_value_cache is not None
    assert enriched.document_store is not None


def test_resolve_host_queue_execution_dependencies_requires_platform_storage() -> None:
    settings = LegalBackendSettings.from_env()
    manifest = build_legal_manifest(settings)
    env = build_legal_environment_profile(settings)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        registry_projection=build_legal_test_registry_projection(settings),
    )
    with pytest.raises(ValueError, match="key_value_cache"):
        resolve_host_queue_execution_dependencies(runtime)


def _build_runtime_with_key_value_cache(
    key_value_cache: object,
):
    settings = LegalBackendSettings.from_env()
    manifest = build_legal_manifest(settings)
    env = build_legal_environment_profile(settings)
    return build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        registry_projection=build_legal_test_registry_projection(settings),
        document_store=InMemoryDocumentStore(),
        key_value_cache=key_value_cache,
    )


def test_resolve_host_queue_execution_dependencies_accepts_direct_distributed_kv_store() -> None:
    runtime = _build_runtime_with_key_value_cache(DispatcherTestKVStore())
    deps = resolve_host_queue_execution_dependencies(runtime)
    assert isinstance(deps.kv_store, DistributedKVStore)


def test_resolve_host_queue_execution_dependencies_accepts_distributed_kv_store_provider() -> None:
    runtime = _build_runtime_with_key_value_cache(
        _QueueWorkerKeyValueCache(kv_store=DispatcherTestKVStore()),
    )
    deps = resolve_host_queue_execution_dependencies(runtime)
    assert isinstance(deps.kv_store, DistributedKVStore)


def test_resolve_host_queue_execution_dependencies_rejects_provider_with_non_kv_store() -> None:
    class _AccidentalKvAttr:
        kv_store = object()

    runtime = _build_runtime_with_key_value_cache(_AccidentalKvAttr())
    with pytest.raises(ValueError, match="DistributedKVStore via kv_store"):
        resolve_host_queue_execution_dependencies(runtime)


def test_resolve_host_queue_execution_dependencies_rejects_unrelated_capability() -> None:
    class _UnrelatedCapability:
        pass

    runtime = _build_runtime_with_key_value_cache(_UnrelatedCapability())
    with pytest.raises(ValueError, match="DistributedKVStoreProvider"):
        resolve_host_queue_execution_dependencies(runtime)


def test_distributed_kv_store_provider_structural_typing_accepts_kv_store_surface() -> None:
    """Runtime-checkable protocol is structural: any object exposing kv_store matches."""
    kv = DispatcherTestKVStore()

    class _KvOnlySurface:
        @property
        def kv_store(self) -> DistributedKVStore:
            return kv

    assert isinstance(_KvOnlySurface(), DistributedKVStoreProvider)


def test_resolve_host_queue_execution_dependencies_wires_platform_storage() -> None:
    settings = LegalBackendSettings.from_env()
    manifest = build_legal_manifest(settings)
    env = build_legal_environment_profile(settings)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        registry_projection=build_legal_test_registry_projection(settings),
        document_store=InMemoryDocumentStore(),
        key_value_cache=_QueueWorkerKeyValueCache(kv_store=DispatcherTestKVStore()),
    )
    deps = resolve_host_queue_execution_dependencies(runtime)
    assert isinstance(deps.kv_store, DistributedKVStore)
    assert deps.causal_evidence_persistence is not None
