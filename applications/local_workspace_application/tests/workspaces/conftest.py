# © Artur Czarnecki. All rights reserved.

"""Shared pytest fixtures for local-workspace application E2Es."""

from __future__ import annotations

import json
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
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

from intergrax.applications._shared.harness_host_runtime import (
    build_harness_host_runtime,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
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

from .rag_e2e_support import (
    _CONNECTION,
    _GRAPH_MAILBOX,
    _GRAPH_TEAM_ID,
    _MARKER_ROOT,
    _NOW,
    _SIGNING_KEY,
    _TENANT,
    _WORKSPACE,
    _RecordingFakeLLM,
    _RecordingSecretsStore,
    _SlackFakeBackend,
)


@pytest.fixture
def rag_e2e_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build an isolated application/RAG environment for each provider E2E."""
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
        slack_tenant_id=_TENANT,
        connected_source_slack_connection_ref=_CONNECTION,
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
        harness_runtime.nexus_loop,
        task_enricher=task_enricher,
        readiness=lifecycle,
    )
    repo = ManagedWorkspaceRepository(store)
    workspace = Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="ws",
        status=WorkspaceStatus.ACTIVE,
        created_at=_NOW,
        updated_at=_NOW,
    )
    repo.put_workspace(workspace)
    repo.put_knowledge_connection_attachment_version_if_absent(
        WorkspaceConnectionAttachment(
            attachment_id="att-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            safe_display_label="Slack",
            status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
            mutation_id="mut-1",
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
    backend = _SlackFakeBackend()
    connection_repository = DocumentStoreTenantConnectionRepository(store)
    TenantConnectionService(
        tenant_id=_TENANT,
        repository=connection_repository,
    ).create(
        TenantConnection(
            connection_ref=_CONNECTION,
            tenant_id=_TENANT,
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
            safe_display_name="Slack",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref="secrets/tenant-a/slack",
            validated_secret_free_config={},
            configuration_version=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    secrets = _RecordingSecretsStore(
        json.dumps(
            {
                "app_token": "xapp-test",
                "bot_token": "xoxb-test",
            }
        )
    )
    factory_registry = build_default_vendor_knowledge_connection_factory_registry(
        slack_runtime_builder=lambda config: SlackConversationChannelIntegration.from_backend(
            backend,
            enabled=True,
            config=config,
        ),
    )
    indexing = WorkspaceDocumentIndexingService(repo, task_executor)
    llm = _RecordingFakeLLM(
        fixed_text=json.dumps(
            {
                "status": "completed",
                "answer": _MARKER_ROOT,
                "used_evidence_ids": ["E1"],
            }
        )
    )
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=task_executor,
        settings=settings,
        repository=repo,
        indexing_service=indexing,
        tenant_connection_secrets_store=secrets,
        tenant_connection_factory_registry=factory_registry,
        msgraph_mailbox_user_id=_GRAPH_MAILBOX,
        msgraph_teams_channel_team_id=_GRAPH_TEAM_ID,
        llm_adapter=llm,
        vectorstore_manager=harness_runtime.env_wiring.tool_wiring.wiring_context.vectorstore_manager,
    )
    with TestClient(app) as client:
        wiring = app.state.lkw_connected_source_wiring
        runtime = app.state.lkw_managed_workspace_sync_runtime
        integration = wiring.connection_registry.resolve(
            tenant_id=_TENANT,
            connection_ref=_CONNECTION,
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        )
        yield {
            "client": client,
            "app": app,
            "backend": backend,
            "wiring": wiring,
            "repo": repo,
            "integration": integration,
            "runtime": runtime,
            "source_catalog": app.state.lkw_tenant_source_catalog,
            "live_catalog": app.state.lkw_tenant_live_capability_catalog,
            "harness_runtime": harness_runtime,
            "settings": settings,
            "llm": llm,
        }
