# © Artur Czarnecki. All rights reserved.

"""HTTP tests for connected Slack workspace knowledge sources."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SlackConversationInventoryPage,
    SlackConversationKind,
    SlackConversationSummary,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving.workspace_routes import mount_managed_workspace_routes
from local_workspace_application.workspaces.connected_source_wiring import (
    build_connected_source_wiring,
    register_slack_connection_integration,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    CreateIndexedSourceMutationHandler,
    DisableIndexedSourceMutationHandler,
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
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_runtime import build_managed_workspace_sync_runtime
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"
_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_CONNECTION = "conn.slack"
_SIGNING_KEY = "http-connected-source-signing-key"
_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


class _FakeBackend:
    async def list_accessible_conversations_page(self, *, cursor, limit):
        return SlackConversationInventoryPage(
            items=(
                SlackConversationSummary(
                    conversation_id="C01234567",
                    kind=SlackConversationKind.PUBLIC_CHANNEL,
                    safe_name="#project-orion",
                    is_archived=False,
                    is_private=False,
                ),
            ),
            next_cursor=None,
        )

    async def read_conversation_history_page(self, **kwargs):
        raise NotImplementedError

    async def read_thread_replies_page(self, **kwargs):
        raise NotImplementedError

    async def read_exact_message(self, **kwargs):
        raise NotImplementedError

    async def read_file_info(self, **kwargs):
        raise NotImplementedError


class _FakeExecutor:
    async def execute(self, task: object) -> object:
        _ = task
        return type(
            "R",
            (),
            {"metadata": {"ingest_summary": {"used": True, "reason": "ingest_complete"}}},
        )()


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        {
            WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE: CreateIndexedSourceMutationHandler(),
            WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE: DisableIndexedSourceMutationHandler(),
        },
    )
    executor = _FakeExecutor()
    indexing = __import__(
        "local_workspace_application.workspaces.document_indexing",
        fromlist=["WorkspaceDocumentIndexingService"],
    ).WorkspaceDocumentIndexingService(repo, executor)
    wiring = build_connected_source_wiring(
        repository=repo,
        workspace_service=service,
        configuration_service=config,
        mutation_engine=mutation_engine,
        indexing_service=indexing,
        settings=settings,
    )
    integration = SlackConversationChannelIntegration.from_backend(
        _FakeBackend(),  # type: ignore[arg-type]
        enabled=True,
    )
    register_slack_connection_integration(
        wiring=wiring,
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        integration=integration,
    )
    sync = ManagedWorkspaceSyncService(
        repo,
        executor,  # type: ignore[arg-type]
        indexing_service=indexing,
        connected_source_sync=wiring.connected_source_sync_service,
    )
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=settings,
        repository=repo,
        sync_runtime=runtime,
        connected_source_wiring=wiring,
        indexing_service=indexing,
    )
    with TestClient(app) as client:
        yield client, wiring


def _headers(*, if_match: str | None = "WKC/0", idempotency: str | None = "idem-1") -> dict[str, str]:
    headers = {"X-Tenant-Id": _TENANT}
    if if_match is not None:
        headers["If-Match"] = if_match
    if idempotency is not None:
        headers["Idempotency-Key"] = idempotency
    return headers


def _candidate_ref(client) -> str:
    discovery = client.get(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/connections/{_CONNECTION}/remote-resources",
        headers={"X-Tenant-Id": _TENANT},
        params={"resource_type": "slack_conversation", "limit": 10},
    )
    assert discovery.status_code == 200, discovery.text
    return discovery.json()["items"][0]["opaque_candidate_ref"]


def _create_indexed_source(client, *, if_match: str = "WKC/1", idempotency: str = "idem-create"):
    response = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
        headers=_headers(if_match=if_match, idempotency=idempotency),
        json={
            "connection_ref": _CONNECTION,
            "opaque_candidate_ref": _candidate_ref(client),
            "root_oldest": "1704067200.000001",
            "root_latest": "1706745600.000001",
        },
    )
    assert response.status_code == 201, response.text
    return response


def test_discovery_route_returns_signed_candidate(api_client) -> None:
    client, _ = api_client
    response = client.get(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/connections/{_CONNECTION}/remote-resources",
        headers={"X-Tenant-Id": _TENANT},
        params={"resource_type": "slack_conversation", "limit": 10},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["items"]
    assert body["items"][0]["safe_display_label"] == "#project-orion"


def test_create_route_requires_preconditions(api_client) -> None:
    client, wiring = api_client
    discovery = client.get(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/connections/{_CONNECTION}/remote-resources",
        headers={"X-Tenant-Id": _TENANT},
        params={"resource_type": "slack_conversation", "limit": 10},
    ).json()
    candidate = discovery["items"][0]["opaque_candidate_ref"]
    missing = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
        headers={"X-Tenant-Id": _TENANT},
        json={
            "connection_ref": _CONNECTION,
            "opaque_candidate_ref": candidate,
            "root_oldest": "1704067200.000001",
            "root_latest": "1706745600.000001",
        },
    )
    assert missing.status_code == 428
    created = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
        headers=_headers(if_match="WKC/1"),
        json={
            "connection_ref": _CONNECTION,
            "opaque_candidate_ref": candidate,
            "root_oldest": "1704067200.000001",
            "root_latest": "1706745600.000001",
        },
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["sync_mode"] == "full"
    assert body["audience_eligibility"] == "personal_only"
    replay = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
        headers=_headers(if_match="WKC/2", idempotency="idem-1"),
        json={
            "connection_ref": _CONNECTION,
            "opaque_candidate_ref": candidate,
            "root_oldest": "1704067200.000001",
            "root_latest": "1706745600.000001",
        },
    )
    assert replay.status_code == 200, replay.text
    conflict = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
        headers=_headers(if_match="WKC/2", idempotency="idem-2"),
        json={
            "connection_ref": _CONNECTION,
            "opaque_candidate_ref": candidate,
            "root_oldest": "1704067200.000001",
            "root_latest": "1706745600.000001",
        },
    )
    assert conflict.status_code == 200, conflict.text


def test_delete_active_indexed_source_returns_disabled(api_client) -> None:
    client, _ = api_client
    created = _create_indexed_source(client)
    binding_id = created.json()["indexed_source_binding_id"]
    disabled = client.delete(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{binding_id}",
        headers=_headers(if_match="WKC/2", idempotency="idem-disable"),
    )
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["status"] == "disabled"


def test_delete_already_disabled_returns_200(api_client) -> None:
    client, _ = api_client
    created = _create_indexed_source(client)
    binding_id = created.json()["indexed_source_binding_id"]
    first = client.delete(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{binding_id}",
        headers=_headers(if_match="WKC/2", idempotency="idem-disable"),
    )
    assert first.status_code == 200, first.text
    second = client.delete(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{binding_id}",
        headers=_headers(if_match="WKC/3", idempotency="idem-disable-2"),
    )
    assert second.status_code == 200, second.text
    assert second.json()["status"] == "disabled"


def test_delete_replay_returns_200(api_client) -> None:
    client, _ = api_client
    created = _create_indexed_source(client)
    binding_id = created.json()["indexed_source_binding_id"]
    first = client.delete(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{binding_id}",
        headers=_headers(if_match="WKC/2", idempotency="idem-disable"),
    )
    assert first.status_code == 200, first.text
    replay = client.delete(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{binding_id}",
        headers=_headers(if_match="WKC/2", idempotency="idem-disable"),
    )
    assert replay.status_code == 200, replay.text
    assert replay.json()["status"] == "disabled"


def test_post_after_disable_reactivates_same_ids(api_client) -> None:
    client, _ = api_client
    created = _create_indexed_source(client)
    body = created.json()
    binding_id = body["indexed_source_binding_id"]
    source_id = body["source_id"]
    disabled = client.delete(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{binding_id}",
        headers=_headers(if_match="WKC/2", idempotency="idem-disable"),
    )
    assert disabled.status_code == 200, disabled.text
    reactivated = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
        headers=_headers(if_match="WKC/3", idempotency="idem-reactivate"),
        json={
            "connection_ref": _CONNECTION,
            "opaque_candidate_ref": _candidate_ref(client),
            "root_oldest": "1704067200.000001",
            "root_latest": "1706745600.000001",
        },
    )
    assert reactivated.status_code == 200, reactivated.text
    revived = reactivated.json()
    assert revived["indexed_source_binding_id"] == binding_id
    assert revived["source_id"] == source_id
    assert revived["status"] == "active"


def test_sync_while_disabled_returns_409(api_client) -> None:
    client, _ = api_client
    created = _create_indexed_source(client)
    binding_id = created.json()["indexed_source_binding_id"]
    disabled = client.delete(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{binding_id}",
        headers=_headers(if_match="WKC/2", idempotency="idem-disable"),
    )
    assert disabled.status_code == 200, disabled.text
    sync = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{binding_id}/sync",
        headers={"X-Tenant-Id": _TENANT},
    )
    assert sync.status_code == 409, sync.text
    assert sync.json()["detail"] == "indexed_source_inactive"


def test_delete_missing_if_match_returns_428(api_client) -> None:
    client, _ = api_client
    created = _create_indexed_source(client)
    binding_id = created.json()["indexed_source_binding_id"]
    response = client.delete(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{binding_id}",
        headers=_headers(if_match=None, idempotency="idem-disable"),
    )
    assert response.status_code == 428, response.text


def test_delete_missing_idempotency_returns_428(api_client) -> None:
    client, _ = api_client
    created = _create_indexed_source(client)
    binding_id = created.json()["indexed_source_binding_id"]
    response = client.delete(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{binding_id}",
        headers=_headers(if_match="WKC/2", idempotency=None),
    )
    assert response.status_code == 428, response.text
