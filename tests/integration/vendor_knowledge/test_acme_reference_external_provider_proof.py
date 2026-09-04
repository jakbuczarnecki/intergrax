# © Artur Czarnecki. All rights reserved.

"""VK-EXT-3 end-to-end qualification for the reference external provider (G3-G7)."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.lkw_task_enricher import (
    build_lkw_combined_task_enricher,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.serving import workspace_routes
from local_workspace_application.serving.workspace_routes import (
    mount_managed_workspace_routes,
)
from local_workspace_application.workspaces.connected_source_host_wiring import (
    build_connected_source_host_bundle,
)
from local_workspace_application.workspaces.connected_source_models import (
    RemoteResourceTypeV1,
)
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    AttachConnectionMutationHandler,
    CreateIndexedSourceMutationHandler,
    DisableIndexedSourceMutationHandler,
)
from local_workspace_application.workspaces.knowledge_connection_detachment_handler import (
    DetachConnectionMutationHandler,
)
from local_workspace_application.workspaces.knowledge_live_access_handlers import (
    CreateLiveAccessBindingMutationHandler,
    DetachLiveAccessBindingMutationHandler,
    DisableLiveAccessBindingMutationHandler,
)
from local_workspace_application.workspaces.knowledge_query_policy_handlers import (
    UpdateQueryPolicyMutationHandler,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService

from applications.local_workspace_application.tests.workspaces.rag_e2e_support import (
    _PREFIX,
    _RecordingFakeLLM,
    _RecordingSecretsStore,
    _SIGNING_KEY,
    _TENANT,
    _WORKSPACE,
    _NOW,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.live.bootstrap import (
    build_vendor_knowledge_live_registration_registry,
)
from intergrax.runtime.vendor_knowledge.provider_composition import (
    build_default_vendor_knowledge_connection_factory_registry,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnection,
    TenantConnectionAdministrativeStatus,
    TenantConnectionService,
)

from acme_reference_vk_plugin.constants import (
    ACME_DEFAULT_COLLECTION_ID,
    ACME_REFERENCE_MARKER,
    ACME_REFERENCE_PROVIDER_ID,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("acme_reference_vk_plugin_installed"),
]

_CONNECTION = "conn.acme-reference"
_CREDENTIAL_REF = "secrets/tenant-a/acme-reference"
_ROOT_OLDEST = "2024-01-01T00:00:00+00:00"
_ROOT_LATEST = "2024-12-31T23:59:59+00:00"


def _wait_operation(client: TestClient, operation_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        response = client.get(
            f"{_PREFIX}/operations/{operation_id}",
            headers={"X-Tenant-Id": _TENANT},
        )
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.05)
    raise AssertionError("operation_timeout")


def _drain_sync(client: TestClient, repo: ManagedWorkspaceRepository, operation_id: str) -> None:
    runtime = client.app.state.lkw_managed_workspace_sync_runtime
    for _ in range(128):
        runtime.worker.drain_once()
        operation = repo.get_operation(tenant_id=_TENANT, operation_id=operation_id)
        if operation is not None and operation.status.value in {"completed", "failed"}:
            return
        time.sleep(0.02)
    raise AssertionError("sync_timeout")


@pytest.fixture
def acme_reference_e2e_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = InMemoryDocumentStore()
    data_home = tmp_path / "lkw-data"
    sqlite_dir = tmp_path / "sqlite"
    shadow_dir = tmp_path / "shadow"
    user_docs = tmp_path / "docs"
    for path in (data_home, sqlite_dir, shadow_dir, user_docs):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_RAG", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_RAG_INGEST", "true")
    monkeypatch.setenv("INTERGRAX_RAG_CHUNKING_STRATEGY", "recursive")
    monkeypatch.setenv("INTERGRAX_RAG_FINAL_TOP_K", "10")
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(user_docs.resolve()))
    monkeypatch.setenv("LKW_DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_SQLITE_DATA_DIR", str(sqlite_dir))
    monkeypatch.setenv("INTERGRAX_SHADOW_ROOT", str(shadow_dir))
    monkeypatch.setenv("LOCAL_WORKSPACE_CONNECTED_SOURCE_OPAQUE_REF_SIGNING_KEY", _SIGNING_KEY)
    monkeypatch.delenv("INTERGRAX_MONGODB_URI", raising=False)
    monkeypatch.setattr(
        workspace_routes,
        "resolve_managed_workspace_document_store",
        lambda document_store=None: store,
    )

    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        data_home=str(data_home),
        connected_source_opaque_ref_signing_key=_SIGNING_KEY,
        slack_tenant_id="",
        connected_source_slack_connection_ref="",
        tenant_connection_bootstrap_tenant_ids=(_TENANT,),
    )
    env = build_local_workspace_environment_profile(settings)
    harness_runtime = build_harness_host_runtime(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        env,
        settings=settings,
        tenant_id=_TENANT,
    )
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.transition_to_ready()
    lifecycle.set_executor_available(True)
    task_enricher = build_lkw_combined_task_enricher(
        env,
        default_capability="local.workspace.search",
        agent_checkpoint_store=harness_runtime.agent_checkpoint_store,
        compensation_queue_store=harness_runtime.compensation_queue_store,
        idempotency_store=harness_runtime.reliability.idempotency_store,
    )
    task_executor = LocalWorkspaceTaskExecutor(
        resolve_harness_host_nexus_loop_legacy(harness_runtime),
        task_enricher=task_enricher,
        readiness=lifecycle,
    )
    repo = ManagedWorkspaceRepository(store)
    workspace_service = ManagedWorkspaceService(repo)
    configuration_service = WorkspaceKnowledgeConfigurationService(repo, workspace_service)
    mutation_engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        workspace_service,
        configuration_service,
        {
            WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE: (
                CreateIndexedSourceMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE: (
                DisableIndexedSourceMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION: (
                AttachConnectionMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION: (
                DetachConnectionMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING: (
                CreateLiveAccessBindingMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.DISABLE_LIVE_ACCESS_BINDING: (
                DisableLiveAccessBindingMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.DETACH_LIVE_ACCESS_BINDING: (
                DetachLiveAccessBindingMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY: (
                UpdateQueryPolicyMutationHandler()
            ),
        },
    )
    repo.put_workspace(
        Workspace(
            workspace_id=_WORKSPACE,
            tenant_id=_TENANT,
            name="acme-reference",
            status=WorkspaceStatus.ACTIVE,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    repo.put_knowledge_connection_attachment_version_if_absent(
        WorkspaceConnectionAttachment(
            attachment_id="att-acme",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            safe_display_label="Acme Reference",
            status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
            mutation_id="mut-acme",
            effective_revision=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    head_mod = __import__(
        "local_workspace_application.workspaces.knowledge_configuration_models",
        fromlist=["WorkspaceKnowledgeConfigurationHead"],
    )
    repo.put_knowledge_configuration_head_if_absent(
        head_mod.WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=1,
            updated_at=_NOW,
        )
    )
    connection_repository = DocumentStoreTenantConnectionRepository(store)
    TenantConnectionService(
        tenant_id=_TENANT,
        repository=connection_repository,
    ).create(
        TenantConnection(
            connection_ref=_CONNECTION,
            tenant_id=_TENANT,
            provider_id=ACME_REFERENCE_PROVIDER_ID,
            integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
            safe_display_name="Acme Reference",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref=_CREDENTIAL_REF,
            validated_secret_free_config={"collection_endpoint": "inmemory://collections"},
            configuration_version=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    secrets = _RecordingSecretsStore(json.dumps({"api_key": "qualification-key"}))
    factory_registry = build_default_vendor_knowledge_connection_factory_registry(
        discover_entry_points=True,
    )
    indexing = WorkspaceDocumentIndexingService(repo, task_executor)
    llm = _RecordingFakeLLM(
        fixed_text=json.dumps(
            {
                "status": "completed",
                "answer": ACME_REFERENCE_MARKER,
                "used_evidence_ids": ["E1"],
            }
        )
    )
    host_bundle = build_connected_source_host_bundle(
        settings=settings,
        repository=repo,
        workspace_service=workspace_service,
        configuration_service=configuration_service,
        mutation_engine=mutation_engine,
        indexing_service=indexing,
        tenant_connection_secrets_store=secrets,
        tenant_connection_factory_registry=factory_registry,
        discover_vendor_knowledge_entry_points=True,
    )
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=task_executor,
        settings=settings,
        repository=repo,
        indexing_service=indexing,
        connected_source_wiring=host_bundle.wiring,
        tenant_connection_secrets_store=secrets,
        tenant_connection_factory_registry=factory_registry,
        llm_adapter=llm,
        vectorstore_manager=harness_runtime.env_wiring.tool_wiring.wiring_context.vectorstore_manager,
    )
    app.state.lkw_tenant_source_catalog = host_bundle.tenant_source_catalog
    with TestClient(app) as client:
        yield {
            "client": client,
            "app": app,
            "repo": repo,
            "wiring": host_bundle.wiring,
            "runtime": app.state.lkw_managed_workspace_sync_runtime,
            "llm": llm,
            "settings": settings,
            "harness_runtime": harness_runtime,
            "secrets": secrets,
            "factory_registry": factory_registry,
            "workspace_service": workspace_service,
            "configuration_service": configuration_service,
            "mutation_engine": mutation_engine,
        }


def _discover_and_bind(client: TestClient) -> tuple[str, str]:
    discovery = client.get(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/connections/{_CONNECTION}/remote-resources",
        headers={"X-Tenant-Id": _TENANT},
        params={
            "resource_type": RemoteResourceTypeV1.VENDOR_KNOWLEDGE_SCOPED_SOURCE.value,
            "limit": 10,
        },
    )
    assert discovery.status_code == 200, discovery.text
    candidate = discovery.json()["items"][0]["opaque_candidate_ref"]
    created = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": "WKC/1",
            "Idempotency-Key": "acme-ext3-create",
        },
        json={
            "connection_ref": _CONNECTION,
            "opaque_candidate_ref": candidate,
            "root_oldest": _ROOT_OLDEST,
            "root_latest": _ROOT_LATEST,
        },
    )
    assert created.status_code == 201, created.text
    body = created.json()
    return body["source_id"], body["indexed_source_binding_id"]


def _sync(client: TestClient, repo: ManagedWorkspaceRepository, binding_id: str) -> None:
    accepted = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{binding_id}/sync",
        headers={"X-Tenant-Id": _TENANT},
    )
    assert accepted.status_code == 202, accepted.text
    operation_id = accepted.json()["operation_id"]
    _drain_sync(client, repo, operation_id)
    completed = _wait_operation(client, operation_id)
    assert completed["status"] == "completed", completed


def test_acme_reference_external_provider_full_proof(acme_reference_e2e_env) -> None:
    client: TestClient = acme_reference_e2e_env["client"]
    repo: ManagedWorkspaceRepository = acme_reference_e2e_env["repo"]
    llm: _RecordingFakeLLM = acme_reference_e2e_env["llm"]
    wiring = acme_reference_e2e_env["wiring"]

    source_id, binding_id = _discover_and_bind(client)
    _sync(client, repo, binding_id)

    search = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": _TENANT},
        json={"query": ACME_REFERENCE_MARKER, "limit": 10},
    )
    assert search.status_code == 200, search.text
    results = search.json()["results"]
    assert results
    assert any(ACME_REFERENCE_MARKER in (hit.get("snippet") or "") for hit in results)
    assert all(hit.get("source_id") == source_id for hit in results)

    ask = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/ask",
        headers={"X-Tenant-Id": _TENANT, "Idempotency-Key": "acme-ext3-ask"},
        json={"question": f"What is the qualification marker {ACME_REFERENCE_MARKER}?"},
    )
    assert ask.status_code == 200, ask.text
    body = ask.json()
    assert body["citations"]
    assert ACME_REFERENCE_MARKER in body["citations"][0]["excerpt"]
    assert any(
        ACME_REFERENCE_MARKER in content
        for message in llm.messages
        for _, content in message
    )

    wrong_tenant = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": "tenant-other"},
        json={"query": ACME_REFERENCE_MARKER, "limit": 10},
    )
    assert wrong_tenant.status_code in {403, 404}

    integration = wiring.connection_registry.resolve(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=ACME_REFERENCE_PROVIDER_ID,
        integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
    )
    assert integration is not None
    assert integration.list_documents(collection_id=ACME_DEFAULT_COLLECTION_ID)


def test_acme_reference_restart_rehydration_search_ask(acme_reference_e2e_env) -> None:
    client: TestClient = acme_reference_e2e_env["client"]
    repo: ManagedWorkspaceRepository = acme_reference_e2e_env["repo"]
    source_id, binding_id = _discover_and_bind(client)
    _sync(client, repo, binding_id)

    old_app: FastAPI = acme_reference_e2e_env["app"]
    settings: LocalWorkspaceBackendSettings = acme_reference_e2e_env["settings"]
    harness_runtime = acme_reference_e2e_env["harness_runtime"]
    secrets = acme_reference_e2e_env["secrets"]
    factory_registry = build_default_vendor_knowledge_connection_factory_registry(
        discover_entry_points=True,
    )
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.transition_to_ready()
    lifecycle.set_executor_available(True)
    env = build_local_workspace_environment_profile(settings)
    task_enricher = build_lkw_combined_task_enricher(
        env,
        default_capability="local.workspace.search",
        agent_checkpoint_store=harness_runtime.agent_checkpoint_store,
        compensation_queue_store=harness_runtime.compensation_queue_store,
        idempotency_store=harness_runtime.reliability.idempotency_store,
    )
    task_executor = LocalWorkspaceTaskExecutor(
        resolve_harness_host_nexus_loop_legacy(harness_runtime),
        task_enricher=task_enricher,
        readiness=lifecycle,
    )
    indexing = WorkspaceDocumentIndexingService(repo, task_executor)
    workspace_service = acme_reference_e2e_env["workspace_service"]
    configuration_service = acme_reference_e2e_env["configuration_service"]
    mutation_engine = acme_reference_e2e_env["mutation_engine"]
    host_bundle = build_connected_source_host_bundle(
        settings=settings,
        repository=repo,
        workspace_service=workspace_service,
        configuration_service=configuration_service,
        mutation_engine=mutation_engine,
        indexing_service=indexing,
        tenant_connection_secrets_store=secrets,
        tenant_connection_factory_registry=factory_registry,
        discover_vendor_knowledge_entry_points=True,
    )
    restarted = FastAPI()
    llm: _RecordingFakeLLM = acme_reference_e2e_env["llm"]
    mount_managed_workspace_routes(
        restarted,
        task_executor=task_executor,
        settings=settings,
        repository=repo,
        indexing_service=indexing,
        connected_source_wiring=host_bundle.wiring,
        tenant_connection_secrets_store=secrets,
        tenant_connection_factory_registry=factory_registry,
        llm_adapter=llm,
        vectorstore_manager=harness_runtime.env_wiring.tool_wiring.wiring_context.vectorstore_manager,
    )
    with TestClient(restarted) as restarted_client:
        search = restarted_client.post(
            f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
            headers={"X-Tenant-Id": _TENANT},
            json={"query": ACME_REFERENCE_MARKER, "limit": 10},
        )
        assert search.status_code == 200, search.text
        assert any(
            ACME_REFERENCE_MARKER in (hit.get("snippet") or "")
            for hit in search.json()["results"]
        )
        ask = restarted_client.post(
            f"{_PREFIX}/workspaces/{_WORKSPACE}/ask",
            headers={"X-Tenant-Id": _TENANT, "Idempotency-Key": "acme-ext3-restart-ask"},
            json={"question": ACME_REFERENCE_MARKER},
        )
        assert ask.status_code == 200, ask.text
        assert ACME_REFERENCE_MARKER in ask.json()["citations"][0]["excerpt"]
    _ = old_app, source_id


def test_builtin_parity_and_live_unchanged_with_external_plugin() -> None:
    from intergrax.runtime.vendor_knowledge.adapter_composition import (
        build_default_vendor_knowledge_adapter_registry,
    )
    from intergrax.runtime.vendor_knowledge.contribution_catalog import (
        build_default_vendor_knowledge_contribution_catalog,
    )
    from intergrax.runtime.vendor_knowledge.plugin_composition import (
        build_default_vendor_knowledge_source_plugin_registry,
    )

    disabled = build_default_vendor_knowledge_contribution_catalog()
    enabled = build_default_vendor_knowledge_contribution_catalog(
        discover_entry_points=True,
    )
    assert len(build_default_vendor_knowledge_adapter_registry().registered_keys()) == 12
    assert len(build_default_vendor_knowledge_source_plugin_registry().list_plugins()) == 12
    assert sum(len(item.connection_factories) for item in disabled.list_contributions()) == 7
    assert sum(len(item.connection_factories) for item in enabled.list_contributions()) == 8
    live_disabled = build_vendor_knowledge_live_registration_registry()
    live_enabled = build_vendor_knowledge_live_registration_registry(
        discover_entry_points=True,
    )
    assert len(live_disabled.list_registrations()) == len(
        live_enabled.list_registrations()
    )
