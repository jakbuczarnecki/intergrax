# © Artur Czarnecki. All rights reserved.

"""End-to-end Slack connected source proof through HTTP, sync, Search and Ask."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
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
from intergrax.integrations.providers.conversation_channel.slack.mapping import parse_slack_ts
from intergrax.runtime.task.task import TaskResult, TaskState
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving.workspace_routes import mount_managed_workspace_routes
from local_workspace_application.workspaces.connected_source_wiring import (
    build_connected_source_wiring,
    register_slack_connection_integration,
)
from local_workspace_application.workspaces.document_indexing import WorkspaceDocumentIndexingResult
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    CreateIndexedSourceMutationHandler,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceDocumentReference,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_runtime import build_managed_workspace_sync_runtime
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService

pytestmark = [pytest.mark.unit]

_MARKER_ROOT = "SLACK-ORION-ROOT-7319"
_MARKER_REPLY = "SLACK-ORION-REPLY-8421"
_MARKER_EDIT = "SLACK-ORION-EDIT-9633"
_CONVERSATION_ID = "C01234567"
_CONNECTION = "conn.slack"
_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_OLDEST = "1704067200.000001"
_LATEST = "1706745600.000001"
_ROOT_TS = "1704153600.000001"
_REPLY_TS = "1704153601.000001"
_EDITED_TS = "1704153602.000001"
_TS = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_PREFIX = "/v1/local_workspace"
_SIGNING_KEY = "e2e-connected-source-signing-key"


class _RecordingFakeLLM(LLMAdapter):
    provider = "fake"
    model = "fake"

    def __init__(self, *, fixed_text: str) -> None:
        super().__init__()
        self._fixed_text = fixed_text

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        _ = messages, temperature, max_tokens, run_id
        return build_adapter_response(content=self._fixed_text)


def _search_task_result(
    *,
    workspace_id: str,
    source_id: str,
    file_name: str,
    document_id: str,
    source_path: str,
    marker: str,
) -> TaskResult:
    return TaskResult(
        task_id="search-1",
        run_id="search-1",
        state=TaskState.COMPLETED,
        answer="ok",
        execution_result=AgentExecutionResult(
            agent_id="local_search",
            run_id="search-1",
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
            structured_data={
                "search_summary": {
                    "query": marker,
                    "workspace_id": workspace_id,
                    "evidence": [
                        {
                            "document_id": document_id,
                            "source_id": source_id,
                            "workspace_id": workspace_id,
                            "source_path": source_path,
                            "file_name": file_name,
                            "snippet": marker,
                            "evidence_id": "E1",
                            "score": 0.99,
                        }
                    ],
                }
            },
        ),
    )


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


class _SlackFakeBackend:
    def __init__(self) -> None:
        self.history_calls = 0
        self._history_pages = [
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_ROOT_TS,
                        text=f"root {_MARKER_ROOT}",
                        reply_count=1,
                    ),
                ),
                next_cursor="history-2",
            ),
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_EDITED_TS,
                        text=f"edited {_MARKER_EDIT}",
                        edited_at=datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc),
                    ),
                ),
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
                        text=f"reply {_MARKER_REPLY}",
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
        page = self._history_pages.pop(0)
        for item in page.items:
            self._content[item.message_ts] = item
        return page

    async def read_thread_replies_page(self, **kwargs: Any) -> SlackConversationMessagePage:
        page = self._reply_pages.pop(0)
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


class _ConnectedSourceIndexingService:
    def __init__(self, repository: ManagedWorkspaceRepository) -> None:
        self._repository = repository
        self._indexed_paths: set[str] = set()

    async def index_one(self, **kwargs: Any) -> WorkspaceDocumentIndexingResult:
        physical_path = Path(str(kwargs["physical_path"]))
        content = physical_path.read_text(encoding="utf-8")
        marker_found = next(
            (m for m in (_MARKER_ROOT, _MARKER_REPLY, _MARKER_EDIT) if m in content),
            None,
        )
        if marker_found is None:
            raise RuntimeError("marker_missing")
        logical_source_path = str(kwargs["logical_source_path"])
        if marker_found not in logical_source_path:
            logical_source_path = f"{logical_source_path}#{marker_found}"
        unchanged = logical_source_path in self._indexed_paths
        if not unchanged:
            self._indexed_paths.add(logical_source_path)
        document_id = f"doc-{len(self._indexed_paths)}"
        if not unchanged:
            self._repository.put_document_ref(
                WorkspaceDocumentReference(
                    document_id=document_id,
                    tenant_id=str(kwargs["tenant_id"]),
                    workspace_id=str(kwargs["workspace_id"]),
                    source_id=str(kwargs["source_id"]),
                    source_path=logical_source_path,
                    file_name=str(kwargs["safe_file_name"]),
                    content_hash=str(kwargs["content_hash"]),
                    indexed_at=datetime.now(UTC),
                )
            )
        return WorkspaceDocumentIndexingResult(
            indexed=not unchanged,
            unchanged=unchanged,
            document_id=document_id,
            documents_indexed=0 if unchanged else 1,
            num_chunks=1,
            reason="ingest_complete",
        )


class _SearchExecutor:
    def __init__(self, repository: ManagedWorkspaceRepository) -> None:
        self._repository = repository

    async def execute(self, task: object) -> TaskResult:
        metadata = getattr(task, "metadata", {}) or {}
        workspace_id = str(metadata.get("workspace_id") or "")
        tenant_id = str(getattr(task, "tenant_id", "") or metadata.get("tenant_id") or "")
        query = str(metadata.get("query") or "")
        refs = self._repository.list_document_refs(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        for ref in refs:
            marker = None
            if _MARKER_ROOT in ref.source_path or _MARKER_ROOT in ref.file_name:
                marker = _MARKER_ROOT
            elif _MARKER_REPLY in ref.source_path or _MARKER_REPLY in ref.file_name:
                marker = _MARKER_REPLY
            elif _MARKER_EDIT in ref.source_path or _MARKER_EDIT in ref.file_name:
                marker = _MARKER_EDIT
            if marker is None:
                continue
            if query and marker not in query and query not in marker:
                continue
            return _search_task_result(
                workspace_id=workspace_id,
                source_id=ref.source_id,
                file_name=ref.file_name,
                document_id=ref.document_id,
                source_path=ref.source_path,
                marker=marker,
            )
        return TaskResult(
            task_id="search-empty",
            run_id="search-empty",
            state=TaskState.COMPLETED,
            answer="ok",
            execution_result=AgentExecutionResult(
                agent_id="local_search",
                run_id="search-empty",
                status=AgentExecutionStatus.COMPLETED,
                summary="ok",
                structured_data={
                    "search_summary": {
                        "query": query,
                        "workspace_id": workspace_id,
                        "evidence": [],
                    }
                },
            ),
        )


@pytest.fixture
def e2e_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("LOCAL_WORKSPACE_CONNECTED_SOURCE_OPAQUE_REF_SIGNING_KEY", _SIGNING_KEY)
    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        data_home=str(data_home),
        connected_source_opaque_ref_signing_key=_SIGNING_KEY,
    )
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
    from local_workspace_application.workspaces.service import ManagedWorkspaceService

    service = ManagedWorkspaceService(repo)
    config = WorkspaceKnowledgeConfigurationService(repo, service)
    mutation_engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        service,
        config,
        {WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE: CreateIndexedSourceMutationHandler()},
    )
    indexing = _ConnectedSourceIndexingService(repo)
    wiring = build_connected_source_wiring(
        repository=repo,
        workspace_service=service,
        configuration_service=config,
        mutation_engine=mutation_engine,
        indexing_service=indexing,  # type: ignore[arg-type]
        settings=settings,
    )
    backend = _SlackFakeBackend()
    integration = SlackConversationChannelIntegration.from_backend(
        backend,  # type: ignore[arg-type]
        enabled=True,
        config=SlackConversationChannelIntegrationConfig(
            enabled=True,
            app_token="xapp-test",
            bot_token="xoxb-test",
        ),
    )
    register_slack_connection_integration(
        wiring=wiring,
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        integration=integration,
    )
    executor = _SearchExecutor(repo)
    llm = _RecordingFakeLLM(
        fixed_text=json.dumps(
            {
                "status": "completed",
                "answer": _MARKER_ROOT,
                "used_evidence_ids": ["E1"],
            }
        )
    )
    sync = ManagedWorkspaceSyncService(
        repo,
        executor,  # type: ignore[arg-type]
        indexing_service=indexing,  # type: ignore[arg-type]
        connected_source_sync=wiring.connected_source_sync_service,
    )
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    wiring.connected_source_sync_service.attach_continuation(
        __import__(
            "local_workspace_application.workspaces.connected_source_wiring",
            fromlist=["_SyncRuntimeContinuation"],
        )._SyncRuntimeContinuation(runtime)
    )
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=settings,
        repository=repo,
        sync_runtime=runtime,
        connected_source_wiring=wiring,
        indexing_service=indexing,  # type: ignore[arg-type]
        llm_adapter=llm,
    )
    with TestClient(app) as client:
        yield client, backend, wiring, repo, integration, runtime


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


def test_slack_connected_source_http_to_search_and_ask(e2e_env) -> None:
    client, backend, wiring, repo, integration, runtime = e2e_env
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
    for _ in range(24):
        if runtime.worker.drain_once() == 0:
            operation = repo.get_operation(tenant_id=_TENANT, operation_id=operation_id)
            if operation is not None and operation.status.value in {"completed", "failed"}:
                break
        time.sleep(0.05)
    completed = _wait_operation(client, operation_id)
    assert completed["status"] == "completed", completed
    assert backend.history_calls >= 1
    assert completed["documents_indexed"] >= 1

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

    missing = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": _TENANT},
        json={"query": "SLACK-ORION-MISSING-0000", "limit": 10},
    )
    assert missing.status_code == 200, missing.text
    assert not any(
        _MARKER_ROOT in (hit.get("snippet") or "")
        for hit in missing.json()["results"]
    )

    ask = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/ask",
        headers={"X-Tenant-Id": _TENANT, "Idempotency-Key": "e2e-ask"},
        json={"question": f"What is {_MARKER_ROOT}?"},
    )
    assert ask.status_code == 200, ask.text
    body = ask.json()
    assert body["citations"]
    assert body["citations"][0]["source_id"] == source_id
    resolved = wiring.connection_registry.resolve(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
    )
    assert resolved is integration
