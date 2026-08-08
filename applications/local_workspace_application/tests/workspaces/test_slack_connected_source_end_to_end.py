# © Artur Czarnecki. All rights reserved.

"""End-to-end Slack connected source proof through HTTP, sync, Search and Ask."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from intergrax.integrations.providers.conversation_channel.slack.backend import (
    SlackConversationChannelBackend,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SlackConversationExactMessageResult,
    SlackConversationInventoryPage,
    SlackConversationKind,
    SlackConversationMessage,
    SlackConversationMessagePage,
    SlackConversationSummary,
    compute_slack_conversation_message_revision,
)
from intergrax.integrations.providers.conversation_channel.slack.mapping import (
    parse_slack_ts,
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
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope

pytestmark = [pytest.mark.unit]

_MARKER_ROOT = "deployment of project Atlas failed because of database timeout"
_MARKER_REPLY = "connection pool exhaustion"
_MARKER_EDIT = "increase pool and add alerting"
_ATLAS_INVESTIGATION = "Atlas deployment investigation identified the database timeout"
_ATLAS_MITIGATION = "Atlas deployment mitigation is to increase the database connection pool"
_UNRELATED_MESSAGE = "routine social update: lunch is at noon"
_CONVERSATION_ID = "C01234567"
_CONNECTION = "conn.slack"
_GRAPH_MAILBOX = "user-abc-123"
_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_OLDEST = "1704067200.000001"
_LATEST = "1706745600.000001"
_ROOT_TS = "1704153600.000001"
_ROOT_2_TS = "1704153602.000001"
_REPLY_TS = "1704153601.000001"
_REPLY_2_TS = "1704153601.000002"
_REPLY_3_TS = "1704153601.000003"
_REPLY_4_TS = "1704153601.000004"
_TS = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_PREFIX = "/v1/local_workspace"
_SIGNING_KEY = "e2e-connected-source-signing-key"


class _RecordingFakeLLM(LLMAdapter):
    provider = "fake"
    model = "fake"

    def __init__(self, *, fixed_text: str) -> None:
        super().__init__()
        self._fixed_text = fixed_text
        self.messages: list[tuple[tuple[str, str], ...]] = []

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        _ = temperature, max_tokens, run_id
        self.messages.append(tuple((message.role, message.content) for message in messages))
        return build_adapter_response(content=self._fixed_text)


class _RecordingSecretsStore:
    def __init__(self, secret: str) -> None:
        self.secret = secret
        self.calls: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        self.calls.append(path)
        return self.secret

    def put_secret(self, path: str, value: str) -> None:
        return None

    def delete_secret(self, path: str) -> None:
        return None


def _assert_slack_rag_citation(
    *,
    citation: dict[str, Any],
    source_id: str,
    marker: str,
) -> None:
    assert citation["source_id"] == source_id
    assert marker in citation["excerpt"]


def _message(
    *,
    message_ts: str,
    text: str,
    reply_count: int = 0,
    root_thread_ts: str | None = None,
    edited_at: datetime | None = None,
) -> SlackConversationMessage:
    created_at = parse_slack_ts(message_ts) or _TS
    return SlackConversationMessage(
        conversation_id=_CONVERSATION_ID,
        message_ts=message_ts,
        root_thread_ts=root_thread_ts,
        actor_provider_id="U111",
        text=text,
        subtype=None,
        created_at=created_at,
        edited_at=edited_at,
        reply_count=reply_count,
        files=(),
        provider_metadata={},
    )


class _SlackFakeBackend(SlackConversationChannelBackend):
    def __init__(self) -> None:
        self.history_calls = 0
        self.reply_calls = 0
        self._history_pages = [
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_ROOT_TS,
                        text=_MARKER_ROOT,
                        reply_count=4,
                    ),
                    _message(
                        message_ts=_ROOT_2_TS,
                        text=_UNRELATED_MESSAGE,
                    ),
                ),
                next_cursor="history-2",
            ),
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(),
            ),
        ]
        self._reply_pages = [
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_REPLY_TS,
                        text=f"{_ATLAS_INVESTIGATION}; {_MARKER_REPLY}",
                        root_thread_ts=_ROOT_TS,
                    ),
                    _message(
                        message_ts=_REPLY_2_TS,
                        text="identified connection pool exhaustion",
                        root_thread_ts=_ROOT_TS,
                    ),
                ),
                next_cursor="replies-2",
            ),
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_REPLY_3_TS,
                        text=_ATLAS_MITIGATION,
                        root_thread_ts=_ROOT_TS,
                    ),
                    _message(
                        message_ts=_REPLY_4_TS,
                        text=f"Atlas deployment final decision: {_MARKER_EDIT}",
                        root_thread_ts=_ROOT_TS,
                    ),
                ),
            )
        ]
        self._content: dict[str, SlackConversationMessage] = {}

    async def list_accessible_conversations_page(self, *, cursor, limit):
        return SlackConversationInventoryPage(
            items=(
                SlackConversationSummary(
                    conversation_id=_CONVERSATION_ID,
                    kind=SlackConversationKind.PUBLIC_CHANNEL,
                    safe_name="#project-orion",
                    is_archived=False,
                    is_private=False,
                ),
            ),
            next_cursor=None,
        )

    async def read_conversation_history_page(self, **kwargs: Any) -> SlackConversationMessagePage:
        self.history_calls += 1
        cursor = kwargs.get("cursor")
        page = self._history_pages[1] if cursor == "history-2" else self._history_pages[0]
        for item in page.items:
            self._content[item.message_ts] = item
        return page

    async def read_thread_replies_page(self, **kwargs: Any) -> SlackConversationMessagePage:
        self.reply_calls += 1
        cursor = kwargs.get("cursor")
        page = self._reply_pages[1] if cursor == "replies-2" else self._reply_pages[0]
        for item in page.items:
            self._content[item.message_ts] = item
        return page

    async def read_exact_message(self, **kwargs: Any) -> SlackConversationExactMessageResult:
        message_ts = kwargs["message_ts"]
        message = self._content.get(message_ts)
        if message is None:
            message = _message(message_ts=message_ts, text="exact")
        revision = kwargs.get("expected_revision")
        if revision is not None and revision != compute_slack_conversation_message_revision(message):
            from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
                SlackConversationMessageChanged,
            )

            raise SlackConversationMessageChanged()
        return SlackConversationExactMessageResult(found=True, message=message)

    async def read_file_info(self, **kwargs: Any):
        raise NotImplementedError


@pytest.fixture
def rag_e2e_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
            backend,  # type: ignore[arg-type]
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
        time.sleep(0.1)
    raise AssertionError("operation_timeout")


def test_slack_connected_source_http_to_search_and_ask(rag_e2e_env) -> None:
    client = rag_e2e_env["client"]
    backend: _SlackFakeBackend = rag_e2e_env["backend"]
    wiring = rag_e2e_env["wiring"]
    repo: ManagedWorkspaceRepository = rag_e2e_env["repo"]
    integration = rag_e2e_env["integration"]
    runtime = rag_e2e_env["runtime"]
    app: FastAPI = rag_e2e_env["app"]
    source_catalog = rag_e2e_env["source_catalog"]
    live_catalog = rag_e2e_env["live_catalog"]
    harness_runtime = rag_e2e_env["harness_runtime"]
    llm: _RecordingFakeLLM = rag_e2e_env["llm"]

    source_capabilities = source_catalog.list_source_kind_capabilities(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
    )
    assert len(source_capabilities) == 1
    assert source_capabilities[0].identity.source_kind == "slack_conversation"
    assert {mode.value for mode in source_capabilities[0].modes} == {
        "DURABLE",
        "INDEXED",
        "LIVE",
    }
    live_capabilities = live_catalog.list_capabilities(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        remote_resource_id=None,
    )
    assert len(live_capabilities) == 3
    assert hasattr(app.state, "lkw_knowledge_inspection_service")
    assert hasattr(app.state, "lkw_knowledge_operations_service")

    discovery = client.get(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/connections/{_CONNECTION}/remote-resources",
        headers={"X-Tenant-Id": _TENANT},
        params={"resource_type": "slack_conversation", "limit": 10},
    )
    assert discovery.status_code == 200, discovery.text
    candidate = discovery.json()["items"][0]["opaque_candidate_ref"]
    created = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": "WKC/1",
            "Idempotency-Key": "e2e-create",
        },
        json={
            "connection_ref": _CONNECTION,
            "opaque_candidate_ref": candidate,
            "root_oldest": _OLDEST,
            "root_latest": _LATEST,
        },
    )
    assert created.status_code == 201, created.text
    source_id = created.json()["source_id"]
    sync_accepted = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{created.json()['indexed_source_binding_id']}/sync",
        headers={"X-Tenant-Id": _TENANT},
    )
    assert sync_accepted.status_code == 202, sync_accepted.text
    operation_id = sync_accepted.json()["operation_id"]
    worker = runtime.worker
    for _ in range(96):
        worker.drain_once()
        operation = repo.get_operation(tenant_id=_TENANT, operation_id=operation_id)
        if operation is not None and operation.status.value in {"completed", "failed"}:
            break
        time.sleep(0.05)
    completed = _wait_operation(client, operation_id)
    assert completed["status"] == "completed", completed
    assert backend.history_calls == 2, completed
    assert backend.reply_calls == 2, completed
    assert completed["documents_indexed"] >= 1

    wiring_ctx = harness_runtime.env_wiring.tool_wiring.wiring_context
    tenant_stores = wiring_ctx.extras.get("tenant_vectorstore_managers", {})
    scoped_manager = tenant_stores.get(_TENANT) or wiring_ctx.vectorstore_manager
    assert scoped_manager is not None
    assert (
        scoped_manager.count(
            scope=VectorStoreScope(tenant_id=_TENANT, workspace_id=_WORKSPACE)
        )
        >= 1
    )

    refs = repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(refs) >= 3
    document_ids = {ref.document_id for ref in refs}
    assert len(document_ids) == len(refs)

    for marker in (_MARKER_ROOT, _MARKER_REPLY, _MARKER_EDIT):
        search = client.post(
            f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
            headers={"X-Tenant-Id": _TENANT},
            json={"query": marker, "limit": 10},
        )
        assert search.status_code == 200, search.text
        results = search.json()["results"]
        assert results
        assert any(marker in (hit.get("snippet") or "") for hit in results)
        assert all(hit.get("source_id") == source_id for hit in results)

    atlas_search = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": _TENANT},
        json={
            "query": "What caused the Atlas deployment failure and what was decided?",
            "limit": 10,
        },
    )
    assert atlas_search.status_code == 200, atlas_search.text
    atlas_results = atlas_search.json()["results"]
    atlas_snippets = [hit["snippet"] for hit in atlas_results]
    assert any(_MARKER_ROOT in snippet for snippet in atlas_snippets)
    assert any(_ATLAS_INVESTIGATION in snippet for snippet in atlas_snippets)
    assert any(_MARKER_REPLY in snippet for snippet in atlas_snippets)
    assert any(_MARKER_EDIT in snippet for snippet in atlas_snippets)
    assert not any(_UNRELATED_MESSAGE in snippet for snippet in atlas_snippets[:1])
    assert any("Message timestamp:" in snippet for snippet in atlas_snippets)
    assert any("Thread root timestamp:" in snippet for snippet in atlas_snippets)
    assert any("Safe locator: slack://" in snippet for snippet in atlas_snippets)

    other_tenant = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": "tenant-other"},
        json={"query": _MARKER_ROOT, "limit": 10},
    )
    assert other_tenant.status_code in {403, 404}

    history_before_ask = backend.history_calls
    ask = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/ask",
        headers={"X-Tenant-Id": _TENANT, "Idempotency-Key": "e2e-ask"},
        json={"question": "What caused the Atlas deployment failure and what was decided?"},
    )
    assert ask.status_code == 200, ask.text
    body = ask.json()
    assert body["citations"]
    _assert_slack_rag_citation(
        citation=body["citations"][0],
        source_id=source_id,
        marker=_MARKER_ROOT,
    )
    assert any(
        _MARKER_ROOT in content
        for message in llm.messages
        for _, content in message
    )
    assert backend.history_calls == history_before_ask
    resolved = wiring.connection_registry.resolve(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
    )
    assert resolved is integration

    retry_sync = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{created.json()['indexed_source_binding_id']}/sync",
        headers={"X-Tenant-Id": _TENANT, "Idempotency-Key": f"retry-{uuid.uuid4().hex}"},
    )
    assert retry_sync.status_code == 202, retry_sync.text
    retry_operation_id = retry_sync.json()["operation_id"]
    for _ in range(48):
        runtime.worker.drain_once()
        operation = repo.get_operation(tenant_id=_TENANT, operation_id=retry_operation_id)
        if operation is not None and operation.status.value in {"completed", "failed"}:
            break
        time.sleep(0.05)
    retry_completed = _wait_operation(client, retry_operation_id)
    assert retry_completed["status"] == "completed", retry_completed
    assert backend.history_calls == 4
    assert backend.reply_calls == 4
    refs_after_retry = repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(refs_after_retry) == len(refs)
    repeated_search = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": _TENANT},
        json={
            "query": "What caused the Atlas deployment failure and what was decided?",
            "limit": 10,
        },
    )
    assert repeated_search.status_code == 200, repeated_search.text
    repeated_ids = sorted(
        hit["document_id"] for hit in repeated_search.json()["results"]
    )
    assert repeated_ids == sorted(hit["document_id"] for hit in atlas_results)

    head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert head is not None
    disabled = client.delete(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/"
        f"{created.json()['indexed_source_binding_id']}",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": f"WKC/{head.committed_revision}",
            "Idempotency-Key": "e2e-disable",
        },
    )
    assert disabled.status_code == 200, disabled.text
