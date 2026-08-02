# © Artur Czarnecki. All rights reserved.

"""Recovery and durability tests for connected workspace source synchronization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
    SlackConversationExactMessageResult,
    SlackConversationInventoryPage,
    SlackConversationKind,
    SlackConversationMessage,
    SlackConversationMessagePage,
    SlackConversationSummary,
    compute_slack_conversation_message_revision,
)
from intergrax.integrations.providers.conversation_channel.slack.mapping import parse_slack_ts
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest
from intergrax.queueing.providers.document_store.document_store_task_queue import (
    DocumentStoreTaskQueue,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SLACK_CONVERSATION_SCOPE_TYPE,
    encode_slack_conversation_scope_id,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
)
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError, VendorKnowledgeErrorCode
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceScope
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
)
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.connected_source_delivery import (
    ConnectedSourceDeliveryApplyResult,
    begin_delivery_receipt,
    complete_delivery_receipt,
    delivery_receipt_completed,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDeliveryReceipt,
    ConnectedSourceDeliveryStatus,
    ConnectedSourceReconciliationStateV1,
    ConnectedSourceSyncSinkError,
)
from local_workspace_application.workspaces.connected_source_recovery import (
    ConnectedSourceRecoveryService,
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
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
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
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_jobs import (
    LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
    ManagedWorkspaceSyncJob,
    encode_managed_workspace_sync_job,
)
from local_workspace_application.workspaces.sync_runtime import build_managed_workspace_sync_runtime
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_SOURCE = "src:connected:test"
_BINDING = "ksb-test"
_INDEXED = "idx-test"
_OPERATION = "op-test"
_OPERATION_OTHER = "op-other"
_DELIVERY = "a" * 64
_CONNECTION = "conn.slack"
_CONVERSATION_ID = "C01234567"
_OLDEST = "1704067200.000001"
_LATEST = "1706745600.000001"
_MARKER = "RECOVERY-MARKER"


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
            document_id="doc-1",
            documents_indexed=0 if unchanged else 1,
        )


@dataclass
class _SyncEnv:
    repo: ManagedWorkspaceRepository
    sync_service: Any
    connected_sync: Any
    backend: _SlackFakeBackend


def _completed_receipt(*, operation_id: str = _OPERATION) -> ConnectedSourceDeliveryReceipt:
    return ConnectedSourceDeliveryReceipt(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
        delivery_id=_DELIVERY,
        binding_configuration_version=1,
        operation_id=operation_id,
        status=ConnectedSourceDeliveryStatus.COMPLETED,
        documents_indexed=1,
        documents_unchanged=0,
        items_failed=0,
        created_at=_NOW,
        completed_at=_NOW,
    )


def _seed_repo(repo: ManagedWorkspaceRepository) -> None:
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
    repo.put_knowledge_indexed_source_version_if_absent(
        WorkspaceIndexedSourceBinding(
            indexed_source_binding_id=_INDEXED,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            knowledge_source_binding_ref=_BINDING,
            source_id=_SOURCE,
            sync_mode=IndexedSourceSyncModeV1.FULL,
            status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
            audience_eligibility=IndexedSourceAudienceEligibilityV1.PERSONAL_ONLY,
            mutation_id="mut-1",
            effective_revision=1,
            semantic_identity_hash="a" * 64,
            created_at=_NOW,
            updated_at=_NOW,
            cached_safe_display_label="#project-orion",
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
    repo.put_source(
        WorkspaceSource(
            source_id=_SOURCE,
            workspace_id=_WORKSPACE,
            tenant_id=_TENANT,
            source_type=WorkspaceSourceType.CONNECTED_SOURCE,
            path="",
            recursive=False,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=_NOW,
            knowledge_configuration_creation_mutation_id="mut-1",
            knowledge_configuration_visibility_revision=1,
        )
    )
    binding_repo = DocumentStoreKnowledgeSourceBindingRepository(repo.document_store)
    encoded_scope = encode_slack_conversation_scope_id(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        oldest=_OLDEST,
        latest=_LATEST,
    )
    binding_repo.create(
        KnowledgeSourceBinding(
            binding_id=_BINDING,
            tenant_id=_TENANT,
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
            source_kind=SLACK_CONVERSATION_SOURCE_KIND,
            connection_ref=_CONNECTION,
            safe_display_name="#project-orion",
            scope=KnowledgeSourceScope(
                remote_scope_id=encoded_scope,
                remote_scope_type=SLACK_CONVERSATION_SCOPE_TYPE,
                safe_display_name="#project-orion",
                parameters={},
            ),
            status=KnowledgeSourceBindingStatus.ACTIVE,
            configuration_version=1,
        )
    )


def _build_sync_env(
    tmp_path: Path,
    *,
    pages: tuple[SlackConversationMessagePage, ...],
    max_pages_per_operation: int = 8,
) -> _SyncEnv:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    settings = LocalWorkspaceBackendSettings(
        data_home=str(tmp_path / "data"),
        connected_source_opaque_ref_signing_key="recovery-signing-key",
        slack_tenant_id=_TENANT,
        connected_source_slack_connection_ref=_CONNECTION,
    )
    from local_workspace_application.workspaces.service import ManagedWorkspaceService

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
    backend = _SlackFakeBackend(pages=pages)
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
    connected_sync._max_pages_per_operation = max_pages_per_operation  # noqa: SLF001
    executor = MagicMock()
    sync_service = ManagedWorkspaceSyncService(
        repo,
        executor,
        indexing_service=indexing,  # type: ignore[arg-type]
        connected_source_sync=connected_sync,
    )
    return _SyncEnv(
        repo=repo,
        sync_service=sync_service,
        connected_sync=connected_sync,
        backend=backend,
    )


def _queued_operation(
    *,
    operation_id: str = _OPERATION,
    reconciliation_state: ConnectedSourceReconciliationStateV1 | None = None,
) -> WorkspaceOperation:
    return WorkspaceOperation(
        operation_id=operation_id,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        operation_type=WorkspaceOperationType.SOURCE_SYNC,
        status=WorkspaceOperationStatus.QUEUED,
        connected_source_reconciliation_state=reconciliation_state,
    )


def test_delivery_identity_without_operation_id_reuses_completed_receipt() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_connected_source_delivery_receipt(_completed_receipt(operation_id=_OPERATION))

    reused = delivery_receipt_completed(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        delivery_id=_DELIVERY,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
        binding_configuration_version=1,
        operation_id=_OPERATION_OTHER,
    )

    assert reused is not None
    assert reused.operation_id == _OPERATION
    stored = repo.get_connected_source_delivery_receipt(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        delivery_id=_DELIVERY,
    )
    assert stored is not None
    assert stored.operation_id == _OPERATION


def test_delivery_receipt_conflict_on_binding_version_mismatch() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_connected_source_delivery_receipt(_completed_receipt())

    with pytest.raises(ConnectedSourceSyncSinkError, match="connected_source_delivery_receipt_conflict"):
        delivery_receipt_completed(
            repository=repo,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            delivery_id=_DELIVERY,
            indexed_source_binding_id=_INDEXED,
            knowledge_source_binding_ref=_BINDING,
            binding_configuration_version=2,
            operation_id=_OPERATION_OTHER,
        )


def test_in_progress_delivery_recovery() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    in_progress = ConnectedSourceDeliveryReceipt(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
        delivery_id=_DELIVERY,
        binding_configuration_version=1,
        operation_id=_OPERATION,
        status=ConnectedSourceDeliveryStatus.IN_PROGRESS,
        created_at=_NOW,
        completed_at=None,
    )
    repo.put_connected_source_delivery_receipt(in_progress)

    assert delivery_receipt_completed(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        delivery_id=_DELIVERY,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
        binding_configuration_version=1,
        operation_id=_OPERATION_OTHER,
    ) is None

    resumed = begin_delivery_receipt(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
        delivery_id=_DELIVERY,
        binding_configuration_version=1,
        operation_id=_OPERATION_OTHER,
    )
    assert resumed.status is ConnectedSourceDeliveryStatus.IN_PROGRESS
    assert resumed.operation_id == _OPERATION


@pytest.mark.asyncio
async def test_sink_complete_then_retryable_vendor_error_requeues(tmp_path: Path) -> None:
    env = _build_sync_env(
        tmp_path,
        pages=(
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(_message(message_ts="1704153600.000001", text=f"root {_MARKER}"),),
                next_cursor=None,
            ),
        ),
    )
    env.repo.put_operation(_queued_operation())
    original_build = env.connected_sync._build_coordinator  # noqa: SLF001

    def _wrap_coordinator(*args: Any, **kwargs: Any):
        coordinator = original_build(*args, **kwargs)
        original_reconcile = coordinator.reconcile_once

        async def _reconcile_once(**reconcile_kwargs: Any) -> KnowledgeSyncRunResult:
            await original_reconcile(**reconcile_kwargs)
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="dependency unavailable",
            )

        coordinator.reconcile_once = _reconcile_once  # type: ignore[method-assign]
        return coordinator

    env.connected_sync._build_coordinator = _wrap_coordinator  # noqa: SLF001

    result = await env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=_OPERATION,
    )

    assert result.status is WorkspaceOperationStatus.QUEUED
    assert result.error == VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE.value
    assert result.documents_indexed == 1
    source = env.repo.get_source(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
    )
    assert source is not None
    assert source.status is WorkspaceSourceStatus.SYNCING


def test_host_interruption_running_to_queued_for_connected_source() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    repo.put_operation(
        WorkspaceOperation(
            operation_id=_OPERATION,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.RUNNING,
            started_at=_NOW,
        )
    )
    repo.put_source(
        repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE).model_copy(  # type: ignore[union-attr]
            update={"status": WorkspaceSourceStatus.SYNCING}
        )
    )
    executor = MagicMock()
    sync_service = ManagedWorkspaceSyncService(repo, executor)
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync_service,
        repository=repo,
        connected_source_recovery_tenant_ids=(_TENANT,),
    )
    job = ManagedWorkspaceSyncJob(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        operation_id=_OPERATION,
    )
    request = TaskRequest(
        tenant_id=_TENANT,
        run_id=_OPERATION,
        task_name=LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
        payload=encode_managed_workspace_sync_job(job),
    )
    runtime.worker._on_interrupted(
        TaskHandle(task_id="task-1", provider="document_store", tenant_id=_TENANT),
        request,
    )  # noqa: SLF001

    reloaded = repo.get_operation(tenant_id=_TENANT, operation_id=_OPERATION)
    assert reloaded is not None
    assert reloaded.status is WorkspaceOperationStatus.QUEUED
    assert reloaded.error is None


@pytest.mark.asyncio
async def test_lease_busy_requeue_leaves_no_orphaned_running(tmp_path: Path) -> None:
    env = _build_sync_env(
        tmp_path,
        pages=(
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(),
                next_cursor=None,
            ),
        ),
    )
    env.repo.put_operation(_queued_operation())

    async def _lease_busy(**kwargs: Any) -> KnowledgeSyncRunResult:
        _ = kwargs
        return KnowledgeSyncRunResult(
            status=KnowledgeSyncRunStatus.LEASE_BUSY,
            mode=KnowledgeSyncMode.RECONCILIATION,
            tenant_id=_TENANT,
            binding_id=_BINDING,
            delivery_id=None,
            changes_count=0,
            active_count=0,
            tombstone_count=0,
            checkpoint_advanced=False,
            has_more=False,
            retryable=True,
        )

    coordinator = MagicMock()
    coordinator.reconcile_once = AsyncMock(side_effect=_lease_busy)
    env.connected_sync._build_coordinator = lambda **kwargs: coordinator  # noqa: SLF001

    result = await env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=_OPERATION,
    )

    assert result.status is WorkspaceOperationStatus.QUEUED
    assert result.error == "lease_busy"
    reloaded = env.repo.get_operation(tenant_id=_TENANT, operation_id=_OPERATION)
    assert reloaded is not None
    assert reloaded.status is WorkspaceOperationStatus.QUEUED


@pytest.mark.asyncio
async def test_durable_counters_no_double_count_on_replay(tmp_path: Path) -> None:
    env = _build_sync_env(
        tmp_path,
        pages=(
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(),
                next_cursor=None,
            ),
        ),
    )
    env.repo.put_operation(_queued_operation())
    env.repo.put_connected_source_delivery_receipt(_completed_receipt())
    apply_results = [
        ConnectedSourceDeliveryApplyResult(
            documents_indexed=1,
            documents_unchanged=0,
            items_processed=1,
            items_failed=0,
            replayed=False,
        ),
        ConnectedSourceDeliveryApplyResult(
            documents_indexed=1,
            documents_unchanged=0,
            items_processed=1,
            items_failed=0,
            replayed=True,
        ),
    ]
    call_index = 0
    original_build = env.connected_sync._build_coordinator  # noqa: SLF001

    def _wrap_coordinator(*args: Any, **kwargs: Any):
        sink = kwargs["sink"]
        coordinator = MagicMock()

        async def _reconcile_once(**reconcile_kwargs: Any) -> KnowledgeSyncRunResult:
            nonlocal call_index
            result = apply_results[min(call_index, len(apply_results) - 1)]
            sink._on_apply(_DELIVERY, result)  # noqa: SLF001
            call_index += 1
            return KnowledgeSyncRunResult(
                status=KnowledgeSyncRunStatus.COMPLETED,
                mode=KnowledgeSyncMode.RECONCILIATION,
                tenant_id=_TENANT,
                binding_id=_BINDING,
                delivery_id=_DELIVERY,
                changes_count=1,
                active_count=1,
                tombstone_count=0,
                checkpoint_advanced=True,
                has_more=False,
                retryable=False,
            )

        coordinator.reconcile_once = _reconcile_once
        return coordinator

    env.connected_sync._build_coordinator = _wrap_coordinator  # noqa: SLF001

    first = await env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=_OPERATION,
    )
    assert first.status is WorkspaceOperationStatus.COMPLETED
    assert first.documents_indexed == 1

    env.repo.put_operation(
        _queued_operation().model_copy(
            update={
                "documents_indexed": first.documents_indexed,
                "documents_unchanged": first.documents_unchanged,
            }
        )
    )
    second = await env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=_OPERATION,
    )
    assert second.status is WorkspaceOperationStatus.COMPLETED
    assert second.documents_indexed == 1


def test_connected_source_recovery_service_requeues_running_operations() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    repo.put_operation(
        WorkspaceOperation(
            operation_id=_OPERATION,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.RUNNING,
            started_at=_NOW,
        )
    )
    repo.put_source(
        repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE).model_copy(  # type: ignore[union-attr]
            update={"status": WorkspaceSourceStatus.ERROR}
        )
    )
    wiring_context = ToolWiringContext(message_bus=DocumentStoreTaskQueue(store))
    recovery = ConnectedSourceRecoveryService(
        repo,
        wiring_context,
        tenant_ids=(_TENANT,),
    )

    result = recovery.recover_running_operations()

    assert result.operations_seen == 1
    assert result.operations_requeued == 1
    reloaded = repo.get_operation(tenant_id=_TENANT, operation_id=_OPERATION)
    assert reloaded is not None
    assert reloaded.status is WorkspaceOperationStatus.QUEUED
    source = repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE)
    assert source is not None
    assert source.status is WorkspaceSourceStatus.SYNCING


def test_stale_operation_audit_does_not_regress_completed_receipt() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    begin_delivery_receipt(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
        delivery_id=_DELIVERY,
        binding_configuration_version=1,
        operation_id=_OPERATION,
    )
    in_progress = repo.get_connected_source_delivery_receipt(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        delivery_id=_DELIVERY,
    )
    assert in_progress is not None
    complete_delivery_receipt(
        repository=repo,
        receipt=in_progress,
        documents_indexed=1,
        documents_unchanged=0,
        items_processed=1,
        items_failed=0,
    )

    stale = begin_delivery_receipt(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
        delivery_id=_DELIVERY,
        binding_configuration_version=1,
        operation_id=_OPERATION_OTHER,
    )
    assert stale.status is ConnectedSourceDeliveryStatus.COMPLETED
    assert stale.operation_id == _OPERATION

    stored = repo.get_connected_source_delivery_receipt(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        delivery_id=_DELIVERY,
    )
    assert stored is not None
    assert stored.status is ConnectedSourceDeliveryStatus.COMPLETED
    assert stored.operation_id == _OPERATION


class _FailingEnqueueBus(DocumentStoreTaskQueue):
    def __init__(self, store: InMemoryDocumentStore, *, fail_times: int = 1) -> None:
        super().__init__(store)
        self._fail_times = fail_times
        self.enqueue_calls = 0

    def enqueue(self, request: TaskRequest) -> TaskHandle:
        self.enqueue_calls += 1
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("enqueue_failed")
        return super().enqueue(request)


def test_requeue_enqueue_failure_repaired_on_recovery() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    failing_bus = _FailingEnqueueBus(store, fail_times=1)
    wiring_context = ToolWiringContext(message_bus=failing_bus)
    from local_workspace_application.workspaces.connected_source_sync_enqueue import (
        durable_requeue_connected_source_operation,
    )

    operation = _queued_operation().model_copy(
        update={"status": WorkspaceOperationStatus.RUNNING}
    )
    repo.put_operation(operation)
    requeued, first_enqueue = durable_requeue_connected_source_operation(
        repository=repo,
        wiring_context=wiring_context,
        operation=operation,
        error_code="lease_busy",
    )
    assert requeued.status is WorkspaceOperationStatus.QUEUED
    assert first_enqueue is not None
    assert first_enqueue.enqueued is False

    recovery = ConnectedSourceRecoveryService(repo, wiring_context, tenant_ids=(_TENANT,))
    result = recovery.recover_running_operations()
    assert result.operations_requeued >= 1
    assert failing_bus.enqueue_calls >= 2


def test_recovery_skips_terminal_operations() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    repo.put_operation(
        _queued_operation().model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "completed_at": _NOW,
            }
        )
    )
    wiring_context = ToolWiringContext(message_bus=DocumentStoreTaskQueue(store))
    recovery = ConnectedSourceRecoveryService(repo, wiring_context, tenant_ids=(_TENANT,))
    result = recovery.recover_running_operations()
    assert result.operations_seen == 0
    assert result.operations_requeued == 0


@pytest.mark.asyncio
async def test_receipt_complete_counter_crash_repaired_on_replay() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_operation(_queued_operation())
    repo.put_connected_source_delivery_receipt(_completed_receipt())
    operation = repo.get_operation(tenant_id=_TENANT, operation_id=_OPERATION)
    assert operation is not None

    from local_workspace_application.workspaces.connected_source_operation_accounting import (
        apply_completed_delivery_accounting,
    )

    replayed = ConnectedSourceDeliveryApplyResult(
        documents_indexed=1,
        documents_unchanged=0,
        items_processed=1,
        items_failed=0,
        replayed=True,
    )
    updated, first = apply_completed_delivery_accounting(
        repository=repo,
        operation=operation,
        delivery_id=_DELIVERY,
        sink_result=replayed,
    )
    assert first.applied is True
    assert updated.documents_indexed == 1

    repaired_again, second = apply_completed_delivery_accounting(
        repository=repo,
        operation=updated,
        delivery_id=_DELIVERY,
        sink_result=replayed,
    )
    assert second.applied is False
    assert repaired_again.documents_indexed == 1
