# © Artur Czarnecki. All rights reserved.

"""Product-level three-mode proof for Slack and Microsoft Graph Teams Chat."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from applications.local_workspace_application.tests.workspaces.test_slack_connected_source_end_to_end import (
    _CONNECTION,
    _NOW,
    _TENANT,
    _WORKSPACE,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MSGRAPH_MAIL_SOURCE_KIND,
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
    MsGraphMailFolder,
    MsGraphMailFolderPage,
    MsGraphMailMessageDeltaPage,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphTeamsChat,
    MsGraphTeamsChatPage,
    MsGraphTeamsChatType,
    MsGraphTeamsIdentity,
    MsGraphTeamsIdentityKind,
)
from intergrax.runtime.vendor_knowledge.live import (
    EffectiveLiveCallBudgetV1,
    KnowledgeQueryAudienceV1,
    LiveExecutionOutcomeV1,
    LiveResultRetentionV1,
)
from intergrax.runtime.vendor_knowledge.live.ms365_graph import (
    MSGRAPH_MAIL_LIST_CAPABILITY_ID,
    MSGRAPH_TEAMS_CHAT_LIST_CAPABILITY_ID,
    MsGraphMailListLiveRequestV1,
    MsGraphTeamsChatListLiveRequestV1,
)
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeMode
from intergrax.runtime.vendor_knowledge.provider_composition import (
    build_default_vendor_knowledge_connection_factory_registry,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionRehydrationStatus,
    TenantConnectionRehydrator,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnection,
    TenantConnectionAdministrativeStatus,
    TenantConnectionService,
)
from local_workspace_application.serving.workspace_routes import (
    KnowledgeConnectionRegistryIntegrationResolverV1,
)
from local_workspace_application.workspaces.hybrid_ask_execution import (
    LiveCapabilityExecutorV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    ExecutableLiveCallV1,
    ResolvedLiveResourceScopeV1,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
)
from tests.unit.runtime.vendor_knowledge.test_msgraph_teams_chat_knowledge_sync import (
    _CHAT_ID,
    _ETAG_1,
    _TeamsChatFakeCollaborationSuite,
    _TeamsChatTestIntegration,
    _active_message,
    _snapshot_page,
)
from tests.unit.runtime.vendor_knowledge.test_msgraph_mail_knowledge_sync import (
    _FOLDER_ID,
    _MailFakeCollaborationSuite,
)

pytest_plugins = (
    "applications.local_workspace_application.tests.workspaces.test_slack_connected_source_end_to_end",
)
pytestmark = pytest.mark.unit

_GRAPH_CONNECTION = "conn.msgraph"
_GRAPH_MARKER = "vk8-graph-marker"
_SLACK_MARKER = "deployment of project Atlas failed because of database timeout"
_GRAPH_SCOPE_START = "2024-01-01T00:00:00+00:00"
_GRAPH_SCOPE_END = "2024-02-01T00:00:00+00:00"
_GRAPH_CREDENTIAL_REF = "secrets/tenant-a/msgraph"


class _CrossProviderGraphClient(_TeamsChatFakeCollaborationSuite):
    def __init__(self) -> None:
        super().__init__()
        message = _active_message(
            remote_id="same-remote-id",
            revision=_ETAG_1,
            subject="Graph VK-8 message",
            body_content=_GRAPH_MARKER,
        ).model_copy(
            update={
                "sender": MsGraphTeamsIdentity(
                    identity_kind=MsGraphTeamsIdentityKind.USER,
                    remote_id="sender-1",
                    display_name="VK8 Graph User",
                    tenant_id=_TENANT,
                )
            }
        )
        self._snapshot_pages = [_snapshot_page(items=(message,))]
        self._snapshot_pages_backup = list(self._snapshot_pages)
        self._content = {("same-remote-id", _ETAG_1): message}

    def read_teams_chats_page(
        self,
        *,
        mailbox_user_id: str,
        continuation,
        limit: int,
    ) -> MsGraphTeamsChatPage:
        _ = continuation, limit
        return MsGraphTeamsChatPage(
            mailbox_user_id=mailbox_user_id,
            items=(
                MsGraphTeamsChat(
                    mailbox_user_id=mailbox_user_id,
                    remote_id=_CHAT_ID,
                    chat_type=MsGraphTeamsChatType.GROUP,
                    topic="Graph VK-8",
                    tenant_id=_TENANT,
                    created_at=datetime(2024, 1, 1, tzinfo=UTC),
                    last_updated_at=datetime(2024, 1, 2, tzinfo=UTC),
                    is_hidden_for_all_members=False,
                    has_online_meeting_info=False,
                ),
            ),
        )


class _MailApplicationGraphClient(_MailFakeCollaborationSuite):
    def _mail_delta_continuation(
        self,
        *,
        mailbox_user_id: str,
        folder_id: str,
    ) -> MsGraphKnowledgeContinuation:
        return MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=(
                "https://graph.microsoft.com/v1.0/users/"
                f"{quote(mailbox_user_id, safe='')}/mailFolders/"
                f"{quote(folder_id, safe='')}/messages/delta?$deltatoken=app-token"
            ),
        )

    def read_mail_folders_page(
        self,
        *,
        mailbox_user_id: str,
        parent_folder_id: str | None,
        continuation,
        limit: int,
    ) -> MsGraphMailFolderPage:
        _ = parent_folder_id, continuation, limit
        return MsGraphMailFolderPage(
            items=(
                MsGraphMailFolder(
                    mailbox_user_id=mailbox_user_id,
                    remote_id=_FOLDER_ID,
                    parent_remote_id=None,
                    display_name="Inbox",
                    child_folder_count=0,
                    total_item_count=3,
                    unread_item_count=1,
                    is_hidden=False,
                ),
            ),
            continuation=None,
        )

    def read_mail_messages_delta_page(
        self,
        *,
        mailbox_user_id: str,
        folder_id: str,
        continuation,
        limit: int,
    ):
        continuation_url = self._mail_delta_continuation(
            mailbox_user_id=mailbox_user_id,
            folder_id=folder_id,
        )
        if continuation is not None:
            return MsGraphMailMessageDeltaPage(items=(), continuation=continuation_url)
        page = super().read_mail_messages_delta_page(
            mailbox_user_id=mailbox_user_id,
            folder_id=folder_id,
            continuation=continuation,
            limit=limit,
        )
        return page.model_copy(
            update={
                "items": tuple(
                    item.model_copy(update={"mailbox_user_id": mailbox_user_id})
                    for item in page.items
                ),
                "continuation": continuation_url,
            }
        )

    def read_mail_message_content(self, *, message, max_chars: int):
        content = super().read_mail_message_content(message=message, max_chars=max_chars)
        return content.model_copy(update={"mailbox_user_id": message.mailbox_user_id})


class _GraphRestartSecretsStore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        _ = version
        self.calls.append(path)
        if path == _GRAPH_CREDENTIAL_REF:
            return "graph-client-secret"
        return json.dumps({"app_token": "xapp-test", "bot_token": "xoxb-test"})

    def put_secret(self, path: str, value: str) -> None:
        _ = path, value

    def delete_secret(self, path: str) -> None:
        _ = path


def _attach_graph_connection(repo) -> None:
    repo.put_knowledge_connection_attachment_version_if_absent(
        WorkspaceConnectionAttachment(
            attachment_id="att-graph",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_GRAPH_CONNECTION,
            safe_display_label="Microsoft Graph",
            status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
            mutation_id="mut-graph",
            effective_revision=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )


def _drain(client: TestClient, runtime, repo, operation_id: str) -> dict[str, object]:
    for _ in range(128):
        runtime.worker.drain_once()
        operation = repo.get_operation(tenant_id=_TENANT, operation_id=operation_id)
        if operation is not None and operation.status.value in {"completed", "failed"}:
            break
        time.sleep(0.02)
    response = client.get(
        f"/v1/local_workspace/operations/{operation_id}",
        headers={"X-Tenant-Id": _TENANT},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _discover(
    client: TestClient,
    *,
    connection: str,
    resource_type: str,
) -> dict[str, object]:
    response = client.get(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/connections/"
        f"{connection}/remote-resources",
        headers={"X-Tenant-Id": _TENANT},
        params={"resource_type": resource_type, "limit": 10},
    )
    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    assert item["resource_type"] == resource_type
    assert item["safe_display_label"]
    assert item["opaque_candidate_ref"]
    return item


def _create(
    client: TestClient,
    *,
    connection: str,
    candidate: str,
    oldest: str,
    latest: str,
    expected_revision: int,
    idempotency: str,
) -> dict[str, object]:
    response = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": f"WKC/{expected_revision}",
            "Idempotency-Key": idempotency,
        },
        json={
            "connection_ref": connection,
            "opaque_candidate_ref": candidate,
            "root_oldest": oldest,
            "root_latest": latest,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["knowledge_source_binding_ref"]
    assert body["source_id"]
    return body


async def _run_live(
    *,
    wiring,
    binding_id: str,
    tenant_id: str = _TENANT,
) -> object:
    binding = wiring.tenant_binding_port.get_binding(
        tenant_id=_TENANT,
        binding_id=binding_id,
    )
    assert binding is not None
    from intergrax.runtime.vendor_knowledge.live.bootstrap import (
        build_vendor_knowledge_live_registration_registry,
    )

    published = build_vendor_knowledge_live_registration_registry().publish()
    call = ExecutableLiveCallV1(
        call_id="vk8-graph-live",
        capability_id=MSGRAPH_TEAMS_CHAT_LIST_CAPABILITY_ID,
        contract_version="1",
        connection_ref=_GRAPH_CONNECTION,
        live_access_binding_id=binding_id,
        remote_resource_id=_CHAT_ID,
        validated_request=MsGraphTeamsChatListLiveRequestV1(page_size=5),
        effective_budget=EffectiveLiveCallBudgetV1(
            max_live_calls=1,
            max_total_duration_ms=30_000,
            max_result_items=5,
            max_result_bytes=32_768,
            max_provider_pages=1,
            max_provider_requests=1,
            max_upstream_items=5,
            max_provider_page_size=5,
            max_content_bytes_per_item=4096,
        ),
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
        resolved_resource_scope=ResolvedLiveResourceScopeV1(
            remote_resource_id=_CHAT_ID,
            scope_token=binding.scope.remote_scope_id,
        ),
    )
    return await LiveCapabilityExecutorV1(
        published_registration=published,
        integration_resolver=KnowledgeConnectionRegistryIntegrationResolverV1(
            wiring.connection_registry
        ),
    ).execute(
        run_id="vk8-graph-live-run",
        tenant_id=tenant_id,
        workspace_id=_WORKSPACE,
        call=call,
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        retention=LiveResultRetentionV1.EPHEMERAL,
    )


async def _run_mail_live(*, wiring, binding_id: str, tenant_id: str = _TENANT) -> object:
    binding = wiring.tenant_binding_port.get_binding(
        tenant_id=_TENANT,
        binding_id=binding_id,
    )
    assert binding is not None
    scope_id = binding.scope.remote_scope_id
    from intergrax.runtime.vendor_knowledge.live.bootstrap import (
        build_vendor_knowledge_live_registration_registry,
    )

    published = build_vendor_knowledge_live_registration_registry().publish()
    call = ExecutableLiveCallV1(
        call_id="mail-lkw-live",
        capability_id=MSGRAPH_MAIL_LIST_CAPABILITY_ID,
        contract_version="1",
        connection_ref=_GRAPH_CONNECTION,
        live_access_binding_id=binding_id,
        remote_resource_id=scope_id,
        validated_request=MsGraphMailListLiveRequestV1(page_size=5),
        effective_budget=EffectiveLiveCallBudgetV1(
            max_live_calls=1,
            max_total_duration_ms=30_000,
            max_result_items=5,
            max_result_bytes=32_768,
            max_provider_pages=1,
            max_provider_requests=1,
            max_upstream_items=5,
            max_provider_page_size=5,
            max_content_bytes_per_item=4096,
        ),
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_MAIL_SOURCE_KIND,
        resolved_resource_scope=ResolvedLiveResourceScopeV1(
            remote_resource_id=scope_id,
            scope_token=scope_id,
        ),
    )
    return await LiveCapabilityExecutorV1(
        published_registration=published,
        integration_resolver=KnowledgeConnectionRegistryIntegrationResolverV1(
            wiring.connection_registry
        ),
    ).execute(
        run_id="mail-lkw-live-run",
        tenant_id=tenant_id,
        workspace_id=_WORKSPACE,
        call=call,
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        retention=LiveResultRetentionV1.EPHEMERAL,
    )


def test_cross_provider_three_mode_e2e(rag_e2e_env) -> None:
    client: TestClient = rag_e2e_env["client"]
    backend = rag_e2e_env["backend"]
    wiring = rag_e2e_env["wiring"]
    repo = rag_e2e_env["repo"]
    runtime = rag_e2e_env["runtime"]
    llm = rag_e2e_env["llm"]

    with pytest.raises(VendorKnowledgeError):
        wiring.connection_registry.resolve(
            tenant_id=_TENANT,
            connection_ref=_GRAPH_CONNECTION,
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )

    connection_repository = DocumentStoreTenantConnectionRepository(repo.document_store)
    TenantConnectionService(
        tenant_id=_TENANT,
        repository=connection_repository,
    ).create(
        TenantConnection(
            connection_ref=_GRAPH_CONNECTION,
            tenant_id=_TENANT,
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            safe_display_name="Microsoft Graph",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref=_GRAPH_CREDENTIAL_REF,
            validated_secret_free_config={
                "client_id": "graph-client-id",
                "default_user": "user-abc-123",
            },
            configuration_version=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    _attach_graph_connection(repo)

    graph_runtime_instances: list[
        tuple[_CrossProviderGraphClient, _TeamsChatTestIntegration]
    ] = []

    def build_graph_runtime(config):
        assert config.client_id == "graph-client-id"
        assert config.client_secret == "graph-client-secret"
        graph_client = _CrossProviderGraphClient()
        graph_integration = _TeamsChatTestIntegration.from_client(
            graph_client,
            enabled=True,
        )
        graph_runtime_instances.append((graph_client, graph_integration))
        return graph_integration

    graph_secrets = _GraphRestartSecretsStore()
    factory_registry = build_default_vendor_knowledge_connection_factory_registry(
        slack_runtime_builder=lambda _config: rag_e2e_env["integration"],
        msgraph_runtime_builder=build_graph_runtime,
    )
    rehydration = TenantConnectionRehydrator(
        repository=connection_repository,
        secrets_store=graph_secrets,
        integration_factory=factory_registry,
        connection_registry=wiring.connection_registry,
    ).rehydrate_tenant(tenant_id=_TENANT)
    graph_rehydration = next(
        result
        for result in rehydration
        if result.connection.connection_ref == _GRAPH_CONNECTION
    )
    assert graph_rehydration.status is TenantConnectionRehydrationStatus.REGISTERED
    assert graph_rehydration.error_code is None
    assert graph_secrets.calls.count(_GRAPH_CREDENTIAL_REF) == 1
    assert len(graph_runtime_instances) == 1
    graph_client, graph_integration = graph_runtime_instances[0]
    assert (
        connection_repository.get(
            tenant_id=_TENANT,
            connection_ref=_GRAPH_CONNECTION,
        ).validated_secret_free_config
        == {"client_id": "graph-client-id", "default_user": "user-abc-123"}
    )
    assert graph_integration is wiring.connection_registry.resolve(
        tenant_id=_TENANT,
        connection_ref=_GRAPH_CONNECTION,
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
    )

    graph_capabilities = rag_e2e_env["source_catalog"].list_source_kind_capabilities(
        tenant_id=_TENANT,
        connection_ref=_GRAPH_CONNECTION,
    )
    graph_teams_chat = next(
        item for item in graph_capabilities
        if item.identity.source_kind == MSGRAPH_TEAMS_CHAT_SOURCE_KIND
    )
    assert set(graph_teams_chat.modes) == {
        VendorKnowledgeMode.DURABLE,
        VendorKnowledgeMode.INDEXED,
        VendorKnowledgeMode.LIVE,
    }

    slack_candidate = _discover(
        client,
        connection=_CONNECTION,
        resource_type="slack_conversation",
    )
    graph_candidate = _discover(
        client,
        connection=_GRAPH_CONNECTION,
        resource_type="teams_chat",
    )

    slack = _create(
        client,
        connection=_CONNECTION,
        candidate=slack_candidate["opaque_candidate_ref"],
        oldest="1704067200.000001",
        latest="1706745600.000001",
        expected_revision=1,
        idempotency="vk8-slack-create",
    )
    graph = _create(
        client,
        connection=_GRAPH_CONNECTION,
        candidate=graph_candidate["opaque_candidate_ref"],
        oldest=_GRAPH_SCOPE_START,
        latest=_GRAPH_SCOPE_END,
        expected_revision=2,
        idempotency="vk8-graph-create",
    )

    assert slack["knowledge_source_binding_ref"] != graph["knowledge_source_binding_ref"]
    assert slack["source_id"] != graph["source_id"]
    assert slack["safe_display_label"] != graph["safe_display_label"]

    slack_sync = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources/"
        f"{slack['indexed_source_binding_id']}/sync",
        headers={"X-Tenant-Id": _TENANT},
    )
    graph_sync = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources/"
        f"{graph['indexed_source_binding_id']}/sync",
        headers={"X-Tenant-Id": _TENANT},
    )
    assert slack_sync.status_code == 202, slack_sync.text
    assert graph_sync.status_code == 202, graph_sync.text
    assert _drain(client, runtime, repo, slack_sync.json()["operation_id"])["status"] == "completed"
    graph_completed = _drain(client, runtime, repo, graph_sync.json()["operation_id"])
    assert graph_completed["status"] == "completed", graph_completed

    refs = repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(refs) >= 4
    assert len({ref.document_id for ref in refs}) == len(refs)
    assert any(
        ref.source_id == slack["source_id"]
        and ref.materialization_ownership is not None
        for ref in refs
    )
    assert any(
        ref.source_id == graph["source_id"]
        and ref.materialization_ownership is not None
        for ref in refs
    )
    healthy_graph_pages = list(graph_client._snapshot_pages_backup)
    graph_client._snapshot_pages = []
    graph_client._snapshot_pages_backup = []
    failed_graph_sync = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources/"
        f"{graph['indexed_source_binding_id']}/sync",
        headers={"X-Tenant-Id": _TENANT},
    )
    assert failed_graph_sync.status_code == 202, failed_graph_sync.text
    runtime.worker.drain_once()
    assert {
        ref.document_id
        for ref in repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    } == {ref.document_id for ref in refs}
    graph_client._snapshot_pages_backup = healthy_graph_pages
    graph_client._snapshot_pages = list(healthy_graph_pages)

    slack_search = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": _TENANT},
        json={"query": _SLACK_MARKER, "limit": 10},
    )
    graph_search = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": _TENANT},
        json={"query": _GRAPH_MARKER, "limit": 10},
    )
    assert slack_search.status_code == graph_search.status_code == 200
    assert all(hit["source_id"] == slack["source_id"] for hit in slack_search.json()["results"])
    assert all(hit["source_id"] == graph["source_id"] for hit in graph_search.json()["results"])

    llm._fixed_text = '{"status":"completed","answer":"graph","used_evidence_ids":["E1"]}'
    graph_ask = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/ask",
        headers={"X-Tenant-Id": _TENANT, "Idempotency-Key": "vk8-graph-ask"},
        json={"question": _GRAPH_MARKER},
    )
    assert graph_ask.status_code == 200, graph_ask.text
    assert graph_ask.json()["citations"]
    assert graph_ask.json()["citations"][0]["source_id"] == graph["source_id"]

    live = asyncio.run(
        _run_live(
            wiring=wiring,
            binding_id=graph["knowledge_source_binding_ref"],
        )
    )
    assert live.normalized_outcome in {
        LiveExecutionOutcomeV1.COMPLETED,
        LiveExecutionOutcomeV1.TRUNCATED,
    }
    assert live.provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert live.source_kind == MSGRAPH_TEAMS_CHAT_SOURCE_KIND
    assert live.item_count <= 5
    assert len(repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)) == len(refs)
    isolated_live = asyncio.run(
        _run_live(
            wiring=wiring,
            binding_id=graph["knowledge_source_binding_ref"],
            tenant_id="tenant-other",
        )
    )
    assert isolated_live.normalized_outcome is LiveExecutionOutcomeV1.FAILED

    other_management = client.delete(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources/"
        f"{graph['indexed_source_binding_id']}",
        headers={
            "X-Tenant-Id": "tenant-other",
            "If-Match": "WKC/3",
            "Idempotency-Key": "vk8-other-tenant-disable",
        },
    )
    assert other_management.status_code in {403, 404}

    inventory = client.app.state.lkw_knowledge_inspection_service.list_items(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    labels = {item.display_label for item in inventory.items}
    assert slack["safe_display_label"] in labels
    assert graph["safe_display_label"] in labels

    stale = client.delete(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources/"
        f"{graph['indexed_source_binding_id']}",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": "WKC/2",
            "Idempotency-Key": "vk8-graph-stale",
        },
    )
    assert stale.status_code == 409, stale.text
    slack_disable = client.delete(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources/"
        f"{slack['indexed_source_binding_id']}",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": "WKC/3",
            "Idempotency-Key": "vk8-slack-disable",
        },
    )
    assert slack_disable.status_code == 200, slack_disable.text
    graph_disable = client.delete(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources/"
        f"{graph['indexed_source_binding_id']}",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": "WKC/4",
            "Idempotency-Key": "vk8-graph-disable",
        },
    )
    assert graph_disable.status_code == 200, graph_disable.text
    inventory_after_disable = client.app.state.lkw_knowledge_inspection_service.list_items(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    graph_item = next(
        item for item in inventory_after_disable.items
        if item.display_label == graph["safe_display_label"]
    )
    assert graph_item.lifecycle_state == "disabled"
    assert "enable" in {
        getattr(action, "value", action) for action in graph_item.available_actions
    }
    other_tenant = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": "tenant-other"},
        json={"query": _GRAPH_MARKER, "limit": 10},
    )
    assert other_tenant.status_code in {403, 404}
    assert backend.history_calls > 0


def test_graph_mail_application_lkw_e2e(rag_e2e_env) -> None:
    client: TestClient = rag_e2e_env["client"]
    wiring = rag_e2e_env["wiring"]
    repo = rag_e2e_env["repo"]
    runtime = rag_e2e_env["runtime"]
    llm = rag_e2e_env["llm"]

    with pytest.raises(VendorKnowledgeError):
        wiring.connection_registry.resolve(
            tenant_id=_TENANT,
            connection_ref=_GRAPH_CONNECTION,
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )

    connection_repository = DocumentStoreTenantConnectionRepository(repo.document_store)
    TenantConnectionService(
        tenant_id=_TENANT,
        repository=connection_repository,
    ).create(
        TenantConnection(
            connection_ref=_GRAPH_CONNECTION,
            tenant_id=_TENANT,
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            safe_display_name="Microsoft Graph",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref=_GRAPH_CREDENTIAL_REF,
            validated_secret_free_config={
                "client_id": "graph-client-id",
                "default_user": "user-abc-123",
            },
            configuration_version=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    _attach_graph_connection(repo)

    graph_runtime_instances: list[_TeamsChatTestIntegration] = []

    def build_graph_runtime(config):
        assert config.client_id == "graph-client-id"
        assert config.client_secret == "graph-client-secret"
        integration = _TeamsChatTestIntegration.from_client(
            _MailApplicationGraphClient(),
            enabled=True,
        )
        graph_runtime_instances.append(integration)
        return integration

    graph_secrets = _GraphRestartSecretsStore()
    factory_registry = build_default_vendor_knowledge_connection_factory_registry(
        slack_runtime_builder=lambda _config: rag_e2e_env["integration"],
        msgraph_runtime_builder=build_graph_runtime,
    )
    rehydration = TenantConnectionRehydrator(
        repository=connection_repository,
        secrets_store=graph_secrets,
        integration_factory=factory_registry,
        connection_registry=wiring.connection_registry,
    ).rehydrate_tenant(tenant_id=_TENANT)
    graph_rehydration = next(
        result
        for result in rehydration
        if result.connection.connection_ref == _GRAPH_CONNECTION
    )
    assert graph_rehydration.status is TenantConnectionRehydrationStatus.REGISTERED
    assert graph_secrets.calls.count(_GRAPH_CREDENTIAL_REF) == 1
    assert len(graph_runtime_instances) == 1

    mail_capabilities = rag_e2e_env["source_catalog"].list_source_kind_capabilities(
        tenant_id=_TENANT,
        connection_ref=_GRAPH_CONNECTION,
    )
    mail = next(
        item for item in mail_capabilities if item.identity.source_kind == MSGRAPH_MAIL_SOURCE_KIND
    )
    assert set(mail.modes) == {
        VendorKnowledgeMode.DURABLE,
        VendorKnowledgeMode.INDEXED,
        VendorKnowledgeMode.LIVE,
    }

    candidate = _discover(
        client,
        connection=_GRAPH_CONNECTION,
        resource_type="mail_folder",
    )
    configuration_head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert configuration_head is not None
    created = _create(
        client,
        connection=_GRAPH_CONNECTION,
        candidate=candidate["opaque_candidate_ref"],
        oldest=_GRAPH_SCOPE_START,
        latest=_GRAPH_SCOPE_END,
        expected_revision=configuration_head.committed_revision,
        idempotency="mail-lkw-create",
    )
    assert created["safe_display_label"] == "Inbox"

    sync = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources/"
        f"{created['indexed_source_binding_id']}/sync",
        headers={"X-Tenant-Id": _TENANT},
    )
    assert sync.status_code == 202, sync.text
    completed = _drain(client, runtime, repo, sync.json()["operation_id"])
    assert completed["status"] == "completed", completed

    refs = repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert refs
    assert any(
        ref.source_id == created["source_id"]
        and ref.materialization_ownership is not None
        for ref in refs
    )
    assert any(
        call["folder_id"] == _FOLDER_ID
        for call in graph_runtime_instances[0]._client.delta_calls
    )

    search = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": _TENANT},
        json={"query": "body-a", "limit": 10},
    )
    assert search.status_code == 200, search.text
    assert search.json()["results"]
    assert all(
        hit["source_id"] == created["source_id"] for hit in search.json()["results"]
    )

    llm._fixed_text = '{"status":"completed","answer":"mail","used_evidence_ids":["E1"]}'
    ask = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/ask",
        headers={"X-Tenant-Id": _TENANT, "Idempotency-Key": "mail-lkw-ask"},
        json={"question": "What is in the mail?"},
    )
    assert ask.status_code == 200, ask.text
    assert ask.json()["citations"]
    assert ask.json()["citations"][0]["source_id"] == created["source_id"]

    live = asyncio.run(
        _run_mail_live(
            wiring=wiring,
            binding_id=created["knowledge_source_binding_ref"],
        )
    )
    assert live.normalized_outcome in {
        LiveExecutionOutcomeV1.COMPLETED,
        LiveExecutionOutcomeV1.TRUNCATED,
    }
    assert live.source_kind == MSGRAPH_MAIL_SOURCE_KIND

    inventory = client.app.state.lkw_knowledge_inspection_service.list_items(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert any(item.display_label == "Inbox" for item in inventory.items)
