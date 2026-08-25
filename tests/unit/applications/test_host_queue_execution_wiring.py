# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.host_queue_execution_wiring import (
    apply_queue_worker_integration_profile,
    resolve_host_queue_execution_dependencies,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
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
