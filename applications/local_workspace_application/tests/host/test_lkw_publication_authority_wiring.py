# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.distributed.source_operation import (
    DocumentStoreSourceOperationCoordinator,
    InProcessSourceOperationCoordinator,
    RagSourceOperationKey,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.tools.providers.rag.source_operation_wiring import (
    bind_source_operation_coordinator,
    shared_source_operation_coordinator,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.workspaces.document_store_factory import (
    resolve_lkw_runtime_document_store,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

pytestmark = pytest.mark.unit

_SCOPE = VectorStoreScope(
    tenant_id="tenant-a",
    namespace="namespace-a",
    workspace_id="workspace-a",
)
_SOURCE_ID = "source-a"


def _build_runtime(document_store: InMemoryDocumentStore):
    settings = LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(settings)
    return build_harness_host_runtime(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        env,
        settings=settings,
        document_store=document_store,
    )


def _source_key() -> RagSourceOperationKey:
    return RagSourceOperationKey(
        tenant_id=_SCOPE.tenant_id,
        namespace=_SCOPE.namespace,
        workspace_id=_SCOPE.workspace_id,
        source_id=_SOURCE_ID,
    )


def test_resolve_lkw_runtime_document_store_uses_inmemory_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("LKW_DATA_HOME", str(data_home))
    monkeypatch.setenv("LOCAL_WORKSPACE_DOCUMENT_STORE_BACKEND", "inmemory")

    settings = LocalWorkspaceBackendSettings.from_env()
    store = resolve_lkw_runtime_document_store(settings)

    assert isinstance(store, InMemoryDocumentStore)


def test_harness_runtime_wires_document_store_into_tool_wiring_context() -> None:
    document_store = InMemoryDocumentStore()
    runtime = _build_runtime(document_store)
    wiring_context = runtime.env_wiring.tool_wiring.wiring_context

    assert wiring_context.document_store is document_store


def test_lkw_document_store_satisfies_conditional_document_store() -> None:
    document_store = InMemoryDocumentStore()
    runtime = _build_runtime(document_store)

    assert isinstance(
        runtime.env_wiring.tool_wiring.wiring_context.document_store,
        ConditionalDocumentStore,
    )


def test_lkw_runtime_selects_document_store_source_operation_coordinator() -> None:
    document_store = InMemoryDocumentStore()
    runtime = _build_runtime(document_store)
    ctx = runtime.env_wiring.tool_wiring.wiring_context

    coordinator = shared_source_operation_coordinator(ctx)

    assert isinstance(coordinator, DocumentStoreSourceOperationCoordinator)
    assert not isinstance(coordinator, InProcessSourceOperationCoordinator)


def test_index_and_retrieve_paths_share_durable_coordinator() -> None:
    document_store = InMemoryDocumentStore()
    runtime = _build_runtime(document_store)
    ctx = runtime.env_wiring.tool_wiring.wiring_context
    manager = VectorstoreManager(object(), scope=_SCOPE)

    bind_source_operation_coordinator(ctx, manager)
    ingest_coordinator = shared_source_operation_coordinator(ctx)
    retrieve_coordinator = shared_source_operation_coordinator(ctx)

    assert ingest_coordinator is retrieve_coordinator
    assert isinstance(ingest_coordinator, DocumentStoreSourceOperationCoordinator)


def test_reconstructed_runtime_recovers_active_publication_generation() -> None:
    document_store = InMemoryDocumentStore()
    before_runtime = _build_runtime(document_store)
    before_ctx = before_runtime.env_wiring.tool_wiring.wiring_context
    before_coordinator = shared_source_operation_coordinator(before_ctx)
    key = _source_key()
    lease = before_coordinator.acquire(key=key)
    assert lease is not None
    assert before_coordinator.promote_publication(lease=lease)
    active_generation = before_coordinator.active_publication_generation(key=key)
    assert active_generation is not None

    after_runtime = _build_runtime(document_store)
    after_ctx = after_runtime.env_wiring.tool_wiring.wiring_context
    after_coordinator = shared_source_operation_coordinator(after_ctx)

    assert isinstance(after_coordinator, DocumentStoreSourceOperationCoordinator)
    assert after_coordinator.active_publication_generation(key=key) == active_generation


def test_factory_repository_and_tool_wiring_share_document_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_home = tmp_path / "factory-data"
    data_home.mkdir()
    monkeypatch.setenv("LKW_DATA_HOME", str(data_home))
    monkeypatch.setenv("LOCAL_WORKSPACE_DOCUMENT_STORE_BACKEND", "inmemory")

    settings = LocalWorkspaceBackendSettings.from_env()
    document_store = resolve_lkw_runtime_document_store(settings)
    repository = ManagedWorkspaceRepository(document_store)
    runtime = build_harness_host_runtime(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        build_local_workspace_environment_profile(settings),
        settings=settings,
        document_store=document_store,
    )

    assert repository.document_store is document_store
    assert (
        runtime.env_wiring.tool_wiring.wiring_context.document_store is document_store
    )


def test_without_document_store_runtime_falls_back_to_inprocess_coordinator() -> None:
    settings = LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(settings)
    runtime = build_harness_host_runtime(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        env,
        settings=settings,
        document_store=None,
    )
    ctx = runtime.env_wiring.tool_wiring.wiring_context

    coordinator = shared_source_operation_coordinator(ctx)

    assert isinstance(coordinator, InProcessSourceOperationCoordinator)


def test_tenant_workspace_source_isolation_in_durable_coordinator() -> None:
    document_store = InMemoryDocumentStore()
    runtime = _build_runtime(document_store)
    coordinator = shared_source_operation_coordinator(
        runtime.env_wiring.tool_wiring.wiring_context,
    )
    source_a = _source_key()
    source_b = RagSourceOperationKey(
        tenant_id="tenant-b",
        namespace=_SCOPE.namespace,
        workspace_id=_SCOPE.workspace_id,
        source_id=_SOURCE_ID,
    )
    lease_a = coordinator.acquire(key=source_a)
    assert lease_a is not None
    assert coordinator.promote_publication(lease=lease_a)

    assert coordinator.active_publication_generation(key=source_a) is not None
    assert coordinator.active_publication_generation(key=source_b) is None
