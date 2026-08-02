# © Artur Czarnecki. All rights reserved.

"""Continuation and checkpoint tests for connected workspace source synchronization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
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
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceReconciliationStateV1,
    RemoteResourceTypeV1,
)
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
from local_workspace_application.workspaces.knowledge_access_service import (
    CreateConnectedIndexedSourceRequest,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceOperationStatus,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_CONNECTION = "conn.slack"
_CONVERSATION_ID = "C01234567"
_OLDEST = "1704067200.000001"
_LATEST = "1706745600.000001"
_SIGNING_KEY = "continuation-signing-key"
_MARKER = "CONTINUATION-MARKER"
_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def _message(*, message_ts: str, text: str) -> SlackConversationMessage:
    created_at = parse_slack_ts(message_ts) or _NOW
    return SlackConversationMessage(
        conversation_id=_CONVERSATION_ID,
        message_ts=message_ts,
        root_thread_ts=None,
        actor_provider_id="U111",
        text=text,
        subtype=None,
        created_at=created_at,
        edited_at=None,
        reply_count=0,
        files=(),
        provider_metadata={},
    )


class _SlackFakeBackend:
    def __init__(self, *, pages: tuple[SlackConversationMessagePage, ...]) -> None:
        self.history_calls = 0
        self._history_pages = list(pages)
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
        return SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(),
        )

    async def read_exact_message(self, **kwargs: Any) -> SlackConversationExactMessageResult:
        message_ts = kwargs["message_ts"]
        message = self._content.get(message_ts) or _message(message_ts=message_ts, text="exact")
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
    def __init__(self) -> None:
        self._indexed_paths: set[str] = set()

    async def index_one(self, **kwargs: Any) -> WorkspaceDocumentIndexingResult:
        physical_path = Path(str(kwargs["physical_path"]))
        content = physical_path.read_text(encoding="utf-8")
        if _MARKER not in content:
            raise RuntimeError("marker_missing")
        logical_source_path = str(kwargs["logical_source_path"])
        unchanged = logical_source_path in self._indexed_paths
        if not unchanged:
            self._indexed_paths.add(logical_source_path)
        return WorkspaceDocumentIndexingResult(
            indexed=not unchanged,
            unchanged=unchanged,
            document_id=f"doc-{len(self._indexed_paths)}",
            documents_indexed=0 if unchanged else 1,
        )


@dataclass
class _ContinuationEnv:
    repo: ManagedWorkspaceRepository
    connected_sync: Any
    backend: _SlackFakeBackend
    operation_id: str
    restart_calls: list[bool]


def _three_history_pages() -> tuple[SlackConversationMessagePage, ...]:
    return (
        SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(_message(message_ts="1704153600.000001", text=f"page-1 {_MARKER}"),),
            next_cursor="history-2",
        ),
        SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(_message(message_ts="1704153601.000001", text=f"page-2 {_MARKER}"),),
            next_cursor="history-3",
        ),
        SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(_message(message_ts="1704153602.000001", text=f"page-3 {_MARKER}"),),
            next_cursor=None,
        ),
    )


@pytest.fixture
def continuation_env(tmp_path: Path) -> _ContinuationEnv:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    data_home = tmp_path / "data"
    data_home.mkdir()
    settings = replace(
        LocalWorkspaceBackendSettings(
            data_home=str(data_home),
            connected_source_opaque_ref_signing_key=_SIGNING_KEY,
            slack_tenant_id=_TENANT,
            connected_source_slack_connection_ref=_CONNECTION,
        ),
        data_home=str(data_home),
    )
    repo.put_workspace(
        Workspace(
            workspace_id=_WORKSPACE,
            tenant_id=_TENANT,
            name="ws",
            status=WorkspaceStatus.ACTIVE,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
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
    service = ManagedWorkspaceService(repo)
    config = WorkspaceKnowledgeConfigurationService(repo, service)
    mutation_engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        service,
        config,
        {
            WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE: (
                CreateIndexedSourceMutationHandler()
            ),
        },
    )
    indexing = _ConnectedSourceIndexingService()
    wiring = build_connected_source_wiring(
        repository=repo,
        workspace_service=service,
        configuration_service=config,
        mutation_engine=mutation_engine,
        indexing_service=indexing,  # type: ignore[arg-type]
        settings=settings,
    )
    backend = _SlackFakeBackend(pages=_three_history_pages())
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
    connected_sync = wiring.connected_source_sync_service
    connected_sync._max_pages_per_operation = 1  # noqa: SLF001

    restart_calls: list[bool] = []
    original_build = connected_sync._build_coordinator  # noqa: SLF001

    def _recording_build(*args: Any, **kwargs: Any):
        coordinator = original_build(*args, **kwargs)
        original_reconcile = coordinator.reconcile_once

        async def _reconcile_once(**reconcile_kwargs: Any):
            restart_calls.append(reconcile_kwargs.get("restart", True))
            return await original_reconcile(**reconcile_kwargs)

        coordinator.reconcile_once = _reconcile_once  # type: ignore[method-assign]
        return coordinator

    connected_sync._build_coordinator = _recording_build  # noqa: SLF001

    async def _create_indexed_source() -> str:
        discovery = await wiring.discovery_service.list_remote_resources(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            resource_type=RemoteResourceTypeV1.SLACK_CONVERSATION,
            limit=10,
            cursor=None,
        )
        candidate = discovery.items[0].opaque_candidate_ref
        created = await wiring.knowledge_access_service.create_indexed_source_from_candidate(
            CreateConnectedIndexedSourceRequest(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                connection_ref=_CONNECTION,
                opaque_candidate_ref=candidate,
                expected_revision=1,
                idempotency_key_hash="b" * 64,
                root_oldest=_OLDEST,
                root_latest=_LATEST,
            )
        )
        operation = service.create_sync_operation(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=created.source_id,
        )
        return operation.operation_id

    operation_id = asyncio.run(_create_indexed_source())
    return _ContinuationEnv(
        repo=repo,
        connected_sync=connected_sync,
        backend=backend,
        operation_id=operation_id,
        restart_calls=restart_calls,
    )


@pytest.mark.asyncio
async def test_new_reconciliation_transitions_to_continuation_after_checkpoint(
    continuation_env: _ContinuationEnv,
) -> None:
    operation = continuation_env.repo.get_operation(
        tenant_id=_TENANT,
        operation_id=continuation_env.operation_id,
    )
    assert operation is not None
    assert operation.connected_source_reconciliation_state is None

    first = await continuation_env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=continuation_env.operation_id,
    )
    assert first.status is WorkspaceOperationStatus.QUEUED
    assert (
        first.connected_source_reconciliation_state
        is ConnectedSourceReconciliationStateV1.CONTINUATION
    )


@pytest.mark.asyncio
async def test_crash_before_first_checkpoint_uses_restart_true(
    continuation_env: _ContinuationEnv,
) -> None:
    await continuation_env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=continuation_env.operation_id,
    )
    assert continuation_env.restart_calls == [True]


@pytest.mark.asyncio
async def test_continuation_after_checkpoint_uses_restart_false(
    continuation_env: _ContinuationEnv,
) -> None:
    await continuation_env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=continuation_env.operation_id,
    )
    continuation_env.restart_calls.clear()

    await continuation_env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=continuation_env.operation_id,
    )
    assert continuation_env.restart_calls == [False]


@pytest.mark.asyncio
async def test_three_pages_with_max_pages_one_per_execution(
    continuation_env: _ContinuationEnv,
) -> None:
    for expected_status in (
        WorkspaceOperationStatus.QUEUED,
        WorkspaceOperationStatus.QUEUED,
        WorkspaceOperationStatus.COMPLETED,
    ):
        operation = await continuation_env.connected_sync.run_operation(
            tenant_id=_TENANT,
            operation_id=continuation_env.operation_id,
        )
        assert operation.status is expected_status

    assert continuation_env.backend.history_calls == 3


@pytest.mark.asyncio
async def test_first_page_read_once_after_checkpoint(
    continuation_env: _ContinuationEnv,
) -> None:
    await continuation_env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=continuation_env.operation_id,
    )
    assert continuation_env.backend.history_calls == 1

    await continuation_env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=continuation_env.operation_id,
    )
    assert continuation_env.backend.history_calls == 2

    await continuation_env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=continuation_env.operation_id,
    )
    assert continuation_env.backend.history_calls == 3


@pytest.mark.asyncio
async def test_checkpoint_commit_before_operation_state_uses_checkpoint_not_first_page(
    continuation_env: _ContinuationEnv,
) -> None:
    await continuation_env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=continuation_env.operation_id,
    )
    assert continuation_env.backend.history_calls == 1
    operation = continuation_env.repo.get_operation(
        tenant_id=_TENANT,
        operation_id=continuation_env.operation_id,
    )
    assert operation is not None
    operation = operation.model_copy(
        update={
            "connected_source_reconciliation_state": None,
            "status": WorkspaceOperationStatus.QUEUED,
        }
    )
    continuation_env.repo.put_operation(operation)

    await continuation_env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=continuation_env.operation_id,
    )
    assert continuation_env.backend.history_calls == 2
    assert continuation_env.restart_calls[-1] is False
