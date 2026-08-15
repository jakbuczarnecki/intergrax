# © Artur Czarnecki. All rights reserved.

"""Recovery and durability tests for connected workspace source synchronization."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.connected_source_delivery import (
    ConnectedSourceDeliveryApplyResult,
    allocate_delivery_sequence,
    begin_delivery_receipt,
    complete_delivery_receipt,
    delivery_receipt_completed,
)
from local_workspace_application.workspaces.connected_source_ids import (
    connected_source_id,
    indexed_source_binding_id,
    workspace_indexed_source_semantic_hash,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDeliveryReceipt,
    ConnectedSourceDeliverySequenceHead,
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
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingResult,
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    CreateIndexedSourceMutationHandler,
    DisableIndexedSourceMutationHandler,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationOutcomeV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
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
from local_workspace_application.workspaces.repository import (
    ManagedWorkspaceRepository,
    WorkspaceKnowledgeConfigurationRepositoryError,
)
from local_workspace_application.workspaces.sync_jobs import (
    LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
    ManagedWorkspaceSyncJob,
    encode_managed_workspace_sync_job,
)
from local_workspace_application.workspaces.sync_runtime import (
    build_managed_workspace_sync_runtime,
)
from local_workspace_application.workspaces.sync_service import (
    ManagedWorkspaceSyncService,
)

from intergrax.compat.langchain.documents import to_langchain_document
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.integrations.providers.conversation_channel.slack.backend import (
    SlackConversationChannelBackend,
)
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
from intergrax.integrations.providers.conversation_channel.slack.mapping import (
    parse_slack_ts,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.knowledge.contracts.validation import _FrozenJsonArray
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest
from intergrax.queueing.providers.document_store.document_store_task_queue import (
    DocumentStoreTaskQueue,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SLACK_CONVERSATION_SCOPE_TYPE,
    encode_slack_conversation_scope_id,
)
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceScope
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
    KnowledgeSyncPublicationFenceV1,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_BINDING = "ksb-test"
_SOURCE = connected_source_id(_TENANT, _WORKSPACE, _BINDING)
_INDEXED = indexed_source_binding_id(_TENANT, _WORKSPACE, _BINDING)
_SEMANTIC = workspace_indexed_source_semantic_hash(_TENANT, _WORKSPACE, _BINDING)
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


class _SlackFakeBackend(SlackConversationChannelBackend):
    def __init__(self, *, pages: tuple[SlackConversationMessagePage, ...]) -> None:
        super().__init__(
            config=SlackConversationChannelIntegrationConfig(
                enabled=True,
                app_token="xapp-test",
                bot_token="xoxb-test",
            )
        )
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

    async def index_connected_source_one(
        self, **kwargs: Any
    ) -> WorkspaceDocumentIndexingResult:
        return await self.index_one(**kwargs)


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
        materialization_sequence=1,
        operation_id=operation_id,
        status=ConnectedSourceDeliveryStatus.COMPLETED,
        documents_indexed=1,
        documents_unchanged=0,
        items_failed=0,
        created_at=_NOW,
        completed_at=_NOW,
    )


def _creation_mutation() -> WorkspaceKnowledgeMutationRecord:
    return WorkspaceKnowledgeMutationRecord(
        mutation_id="mut-1",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
        idempotency_key_hash="f" * 64,
        normalized_request_hash="a" * 64,
        semantic_identity_hash=_SEMANTIC,
        target_revision=1,
        committed_revision=1,
        status=WorkspaceKnowledgeMutationStatusV1.COMMITTED,
        outcome=WorkspaceKnowledgeMutationOutcomeV1.APPLIED,
        result_entity_type="indexed_source_binding",
        result_entity_id=_INDEXED,
        created_at=_NOW,
        updated_at=_NOW,
        committed_at=_NOW,
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
            semantic_identity_hash=_SEMANTIC,
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
    repo.put_knowledge_configuration_mutation_if_absent(_creation_mutation())
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
    publication_fence = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        repo.document_store
    )
    publication_fence.write_fence(
        KnowledgeSyncPublicationFenceV1(
            tenant_id=_TENANT,
            binding_id=_BINDING,
            lifecycle_revision=1,
            lifecycle_token="recovery-test-token",
            enabled=True,
            detached=False,
        ),
        expected_revision=None,
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
            WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE: (
                DisableIndexedSourceMutationHandler()
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
    connected_sync._max_pages_per_operation = max_pages_per_operation
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


def test_delivery_sequence_allocation_is_durable_and_cas_ordered() -> None:
    store = InMemoryDocumentStore()
    first_repository = ManagedWorkspaceRepository(store)
    second_repository = ManagedWorkspaceRepository(store)
    delivery_b = "b" * 64

    def _allocate(args: tuple[ManagedWorkspaceRepository, str]) -> int:
        repository, delivery_id = args
        return allocate_delivery_sequence(
            repository=repository,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_INDEXED,
            delivery_id=delivery_id,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(_allocate, (first_repository, _DELIVERY)),
            executor.submit(_allocate, (second_repository, delivery_b)),
        )
        sequences = [future.result() for future in futures]

    sequence_by_delivery = {
        _DELIVERY: sequences[0],
        delivery_b: sequences[1],
    }
    assert sorted(sequence_by_delivery.values()) == [1, 2]
    replay_sequences = [
        allocate_delivery_sequence(
            repository=repository,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_INDEXED,
            delivery_id=_DELIVERY,
        )
        for repository in (first_repository, second_repository)
    ]
    assert replay_sequences == [sequence_by_delivery[_DELIVERY]] * 2

    recovered = begin_delivery_receipt(
        repository=second_repository,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
        delivery_id=_DELIVERY,
        binding_configuration_version=1,
        operation_id=_OPERATION,
    )
    assert recovered.materialization_sequence == sequence_by_delivery[_DELIVERY]

    next_delivery = begin_delivery_receipt(
        repository=first_repository,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
        delivery_id="c" * 64,
        binding_configuration_version=1,
        operation_id=_OPERATION,
    )
    assert next_delivery.materialization_sequence == 3

    racing_replay_id = "d" * 64
    with ThreadPoolExecutor(max_workers=2) as executor:
        replay_futures = (
            executor.submit(_allocate, (first_repository, racing_replay_id)),
            executor.submit(_allocate, (second_repository, racing_replay_id)),
        )
        racing_replay_sequences = [future.result() for future in replay_futures]
    assert racing_replay_sequences == [4, 4]
    head = first_repository.get_connected_source_delivery_sequence_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
    )
    assert head is not None
    assert set(head.model_dump()) == {
        "tenant_id",
        "workspace_id",
        "source_id",
        "indexed_source_binding_id",
        "next_sequence",
    }
    assert "delivery_sequences" not in head.model_dump()
    assignments = first_repository.list_connected_source_delivery_sequence_assignments(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
    )
    assert len(assignments) == 4
    assert len({item.delivery_id for item in assignments}) == 4
    assert len({item.materialization_sequence for item in assignments}) == 4


def test_delivery_sequence_storage_shape_scales_without_growing_head() -> None:
    repository = ManagedWorkspaceRepository(InMemoryDocumentStore())
    delivery_ids = tuple(f"{index:064x}" for index in range(750))

    for delivery_id in delivery_ids:
        assert (
            allocate_delivery_sequence(
                repository=repository,
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                source_id=_SOURCE,
                indexed_source_binding_id=_INDEXED,
                delivery_id=delivery_id,
            )
            > 0
        )

    head = repository.get_connected_source_delivery_sequence_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
    )
    assert isinstance(head, ConnectedSourceDeliverySequenceHead)
    assert head.next_sequence == 751
    assert "delivery_sequences" not in head.model_dump()
    assignments = repository.list_connected_source_delivery_sequence_assignments(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
    )
    assert len(assignments) == 750
    assert {item.delivery_id for item in assignments} == set(delivery_ids)
    assert {item.materialization_sequence for item in assignments} == set(range(1, 751))


def test_delivery_sequence_crash_gap_is_recoverable() -> None:
    repository = ManagedWorkspaceRepository(InMemoryDocumentStore())
    first_delivery = allocate_delivery_sequence(
        repository=repository,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        delivery_id=_DELIVERY,
    )
    head = repository.get_connected_source_delivery_sequence_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
    )
    assert head is not None
    assert repository.replace_connected_source_delivery_sequence_head_if_match(
        expected=head,
        replacement=head.model_copy(update={"next_sequence": head.next_sequence + 1}),
    )

    recovered = allocate_delivery_sequence(
        repository=repository,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        delivery_id="b" * 64,
    )
    assert first_delivery == 1
    assert recovered == 3
    assert (
        allocate_delivery_sequence(
            repository=repository,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_INDEXED,
            delivery_id="b" * 64,
        )
        == 3
    )


def test_old_delivery_sequence_ledger_requires_explicit_migration() -> None:
    store = InMemoryDocumentStore()
    repository = ManagedWorkspaceRepository(store)
    store.put(
        DocumentRecord(
            partition_key=f"lkw.managed_workspace:{_TENANT}:connected_source_delivery_sequence",
            row_key=f"{_WORKSPACE}:{_SOURCE}:{_INDEXED}",
            data={
                "tenant_id": _TENANT,
                "workspace_id": _WORKSPACE,
                "source_id": _SOURCE,
                "indexed_source_binding_id": _INDEXED,
                "next_sequence": 2,
                "delivery_sequences": {_DELIVERY: 1},
            },
        )
    )

    with pytest.raises(
        WorkspaceKnowledgeConfigurationRepositoryError,
        match="connected_source_delivery_sequence_migration_required",
    ):
        repository.get_connected_source_delivery_sequence_head(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_INDEXED,
        )


def test_langchain_transport_normalizes_frozen_nested_metadata() -> None:
    metadata = {
        "nested": {
            "ordered": [
                "first",
                {"provenance": {"events": ["created", "indexed"]}},
                3,
            ]
        }
    }
    native = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "slack:message-1",
                "root_document_id": "slack:message-1",
            },
            "scope": {"tenant_id": _TENANT, "namespace": "workspace"},
            "content": "Slack message",
            "metadata": metadata,
            "provenance": {
                "source_kind": "slack_message",
                "source_id": _SOURCE,
                "source_parent_id": "conversation:C01234567",
                "provider_id": SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
                "source_revision": "rev-1",
                "source_uri": "slack://C01234567/1704153600.000001",
                "content_hash": "a" * 64,
            },
        }
    )

    assert isinstance(native.metadata["nested"]["ordered"], _FrozenJsonArray)

    transported = to_langchain_document(native)
    serialized = json.loads(transported.model_dump_json())

    assert serialized["metadata"]["nested"]["ordered"] == [
        "first",
        {"provenance": {"events": ["created", "indexed"]}},
        3,
    ]
    assert serialized["metadata"]["source_kind"] == "slack_message"
    assert serialized["metadata"]["source_id"] == _SOURCE
    assert serialized["metadata"]["source_parent_id"] == "conversation:C01234567"
    assert serialized["metadata"]["provider_id"] == SLACK_CONVERSATION_CHANNEL_PROVIDER_ID
    assert serialized["metadata"]["source_revision"] == "rev-1"
    assert serialized["metadata"]["source_uri"] == "slack://C01234567/1704153600.000001"
    assert serialized["metadata"]["content_hash"] == "a" * 64
    assert metadata == {
        "nested": {
            "ordered": [
                "first",
                {"provenance": {"events": ["created", "indexed"]}},
                3,
            ]
        }
    }


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
        materialization_sequence=1,
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
    original_build = env.connected_sync._build_coordinator

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

    env.connected_sync._build_coordinator = _wrap_coordinator

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


@pytest.mark.asyncio
async def test_materialization_receipt_retry_does_not_reread_provider(tmp_path: Path, monkeypatch) -> None:
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
    import local_workspace_application.workspaces.connected_source_sync_service as sync_module

    original_accounting = sync_module.apply_completed_delivery_accounting
    fail_once = True

    def _crash_after_materialization(*args: Any, **kwargs: Any):
        nonlocal fail_once
        result = original_accounting(*args, **kwargs)
        if fail_once:
            fail_once = False
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="crash after materialization receipt",
            )
        return result

    monkeypatch.setattr(
        sync_module,
        "apply_completed_delivery_accounting",
        _crash_after_materialization,
    )
    first = await env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=_OPERATION,
    )
    assert first.status is WorkspaceOperationStatus.QUEUED
    history_calls = env.backend.history_calls
    refs_after_materialization = env.repo.list_document_refs(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )

    second = await env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=_OPERATION,
    )
    assert second.status is WorkspaceOperationStatus.COMPLETED
    assert env.backend.history_calls == history_calls
    assert env.repo.list_document_refs(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    ) == refs_after_materialization


@pytest.mark.asyncio
async def test_checkpoint_committed_retry_is_noop_without_provider_reread(
    tmp_path: Path,
) -> None:
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
    original_build = env.connected_sync._build_coordinator
    fail_once = True

    def _build_with_fail_once(*args: Any, **kwargs: Any):
        coordinator = original_build(*args, **kwargs)
        original_reconcile = coordinator.reconcile_once

        async def _reconcile_once(**reconcile_kwargs: Any):
            nonlocal fail_once
            result = await original_reconcile(**reconcile_kwargs)
            if fail_once:
                fail_once = False
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                    safe_message="crash after checkpoint commit",
                )
            return result

        coordinator.reconcile_once = _reconcile_once  # type: ignore[method-assign]
        return coordinator

    env.connected_sync._build_coordinator = _build_with_fail_once
    first = await env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=_OPERATION,
    )
    assert first.status is WorkspaceOperationStatus.QUEUED
    history_calls = env.backend.history_calls

    second = await env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=_OPERATION,
    )
    assert second.status is WorkspaceOperationStatus.COMPLETED
    assert env.backend.history_calls == history_calls


@pytest.mark.asyncio
async def test_completed_index_receipt_replays_without_duplicate_index_effect(
    tmp_path: Path,
) -> None:
    class _IndexExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, task):
            self.calls += 1
            return type(
                "Result",
                (),
                {"metadata": {"ingest_summary": {"used": True, "num_chunks": 2}}},
            )()

    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    executor = _IndexExecutor()
    service = WorkspaceDocumentIndexingService(repo, executor)
    document_path = tmp_path / "slack.md"
    document_path.write_text("marker", encoding="utf-8")
    original_put = repo.put_document_ref
    fail_once = True

    def _crash_before_lifecycle_advance(reference):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("crash after index receipt")
        return original_put(reference)

    repo.put_document_ref = _crash_before_lifecycle_advance  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="crash after index receipt"):
        await service.index_one(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            operation_id=_OPERATION,
            physical_path=document_path,
            logical_source_path="connected/slack-message/one.md",
            safe_file_name="slack.md",
            content_hash="a" * 64,
        )
    repo.put_document_ref = original_put  # type: ignore[method-assign]

    replayed = await service.index_one(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        operation_id=_OPERATION,
        physical_path=document_path,
        logical_source_path="connected/slack-message/one.md",
        safe_file_name="slack.md",
        content_hash="a" * 64,
    )
    assert replayed.reason == "index_replayed"
    assert replayed.documents_indexed == 1
    assert executor.calls == 1


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
    )

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
    env.connected_sync._build_coordinator = lambda **kwargs: coordinator

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

    def _wrap_coordinator(*args: Any, **kwargs: Any):
        sink = kwargs["sink"]
        coordinator = MagicMock()

        async def _reconcile_once(**reconcile_kwargs: Any) -> KnowledgeSyncRunResult:
            nonlocal call_index
            result = apply_results[min(call_index, len(apply_results) - 1)]
            sink._on_apply(_DELIVERY, result)
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

    env.connected_sync._build_coordinator = _wrap_coordinator

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


def test_accounting_record_exists_zero_counters_repaired_on_retry() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_operation(_queued_operation())
    repo.put_connected_source_delivery_receipt(_completed_receipt())
    from local_workspace_application.workspaces.connected_source_models import (
        ConnectedSourceOperationDeliveryAccounting,
    )
    from local_workspace_application.workspaces.connected_source_operation_accounting import (
        apply_completed_delivery_accounting,
    )

    repo.put_connected_source_delivery_accounting_if_absent(
        ConnectedSourceOperationDeliveryAccounting(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_INDEXED,
            knowledge_source_binding_ref=_BINDING,
            operation_id=_OPERATION,
            delivery_id=_DELIVERY,
            documents_indexed=1,
            documents_unchanged=0,
            items_failed=0,
            accounted_at=_NOW,
            ownership_classification="COMPLETE_OWNERSHIP",
        )
    )
    operation = repo.get_operation(tenant_id=_TENANT, operation_id=_OPERATION)
    assert operation is not None
    assert operation.documents_indexed == 0

    updated, result = apply_completed_delivery_accounting(
        repository=repo,
        operation=operation,
        delivery_id=_DELIVERY,
    )
    assert result.applied is True
    assert updated.documents_indexed == 1


def test_accounting_second_retry_counters_unchanged() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_operation(_queued_operation().model_copy(update={"documents_indexed": 1}))
    repo.put_connected_source_delivery_receipt(_completed_receipt())
    from local_workspace_application.workspaces.connected_source_models import (
        ConnectedSourceOperationDeliveryAccounting,
    )
    from local_workspace_application.workspaces.connected_source_operation_accounting import (
        apply_completed_delivery_accounting,
    )

    repo.put_connected_source_delivery_accounting_if_absent(
        ConnectedSourceOperationDeliveryAccounting(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_INDEXED,
            knowledge_source_binding_ref=_BINDING,
            operation_id=_OPERATION,
            delivery_id=_DELIVERY,
            documents_indexed=1,
            documents_unchanged=0,
            items_failed=0,
            accounted_at=_NOW,
            ownership_classification="COMPLETE_OWNERSHIP",
        )
    )
    operation = repo.get_operation(tenant_id=_TENANT, operation_id=_OPERATION)
    assert operation is not None
    _, second = apply_completed_delivery_accounting(
        repository=repo,
        operation=operation,
        delivery_id=_DELIVERY,
    )
    assert second.applied is False
    assert operation.documents_indexed == 1


def test_accounting_two_deliveries_aggregate_exact_counters() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_operation(_queued_operation())
    delivery_b = "b" * 64
    repo.put_connected_source_delivery_receipt(_completed_receipt())
    repo.put_connected_source_delivery_receipt(
        _completed_receipt().model_copy(
            update={"delivery_id": delivery_b, "documents_indexed": 2, "documents_unchanged": 1}
        )
    )
    from local_workspace_application.workspaces.connected_source_models import (
        ConnectedSourceOperationDeliveryAccounting,
    )
    from local_workspace_application.workspaces.connected_source_operation_accounting import (
        apply_completed_delivery_accounting,
    )

    for delivery_id, indexed, unchanged in (
        (_DELIVERY, 1, 0),
        (delivery_b, 2, 1),
    ):
        repo.put_connected_source_delivery_accounting_if_absent(
            ConnectedSourceOperationDeliveryAccounting(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                source_id=_SOURCE,
                indexed_source_binding_id=_INDEXED,
                knowledge_source_binding_ref=_BINDING,
                operation_id=_OPERATION,
                delivery_id=delivery_id,
                documents_indexed=indexed,
                documents_unchanged=unchanged,
                items_failed=0,
                accounted_at=_NOW,
                ownership_classification="COMPLETE_OWNERSHIP",
            )
        )
    operation = repo.get_operation(tenant_id=_TENANT, operation_id=_OPERATION)
    assert operation is not None
    updated, result = apply_completed_delivery_accounting(
        repository=repo,
        operation=operation,
        delivery_id=_DELIVERY,
    )
    assert result.applied is True
    assert updated.documents_indexed == 3
    assert updated.documents_unchanged == 1


def test_conflicting_accounting_record_raises_deterministic_failure() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_operation(_queued_operation())
    repo.put_connected_source_delivery_receipt(_completed_receipt())
    from local_workspace_application.workspaces.connected_source_models import (
        ConnectedSourceOperationDeliveryAccounting,
    )
    from local_workspace_application.workspaces.connected_source_operation_accounting import (
        ConnectedSourceDeliveryAccountingConflictError,
        apply_completed_delivery_accounting,
    )

    repo.put_connected_source_delivery_accounting_if_absent(
        ConnectedSourceOperationDeliveryAccounting(
            tenant_id=_TENANT,
            operation_id=_OPERATION,
            delivery_id=_DELIVERY,
            documents_indexed=9,
            documents_unchanged=0,
            items_failed=0,
            accounted_at=_NOW,
        )
    )
    operation = repo.get_operation(tenant_id=_TENANT, operation_id=_OPERATION)
    assert operation is not None
    with pytest.raises(ConnectedSourceDeliveryAccountingConflictError) as exc_info:
        apply_completed_delivery_accounting(
            repository=repo,
            operation=operation,
            delivery_id=_DELIVERY,
        )
    assert exc_info.value.error_code == "connected_source_delivery_accounting_conflict"


def test_stale_operation_cannot_overwrite_terminal_status() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_operation(
        _queued_operation().model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "completed_at": _NOW,
            }
        )
    )
    repo.put_connected_source_delivery_receipt(_completed_receipt())
    from local_workspace_application.workspaces.connected_source_operation_accounting import (
        apply_completed_delivery_accounting,
    )

    stale = _queued_operation()
    updated, result = apply_completed_delivery_accounting(
        repository=repo,
        operation=stale,
        delivery_id=_DELIVERY,
    )
    assert result.applied is True
    assert updated.status is WorkspaceOperationStatus.COMPLETED


def test_enqueue_generation_allocation_is_monotonic() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    operation = _queued_operation()
    first = repo.allocate_connected_source_sync_enqueue_generation(
        tenant_id=operation.tenant_id,
        workspace_id=operation.workspace_id,
        source_id=operation.source_id,
        operation_id=operation.operation_id,
    )
    second = repo.allocate_connected_source_sync_enqueue_generation(
        tenant_id=operation.tenant_id,
        workspace_id=operation.workspace_id,
        source_id=operation.source_id,
        operation_id=operation.operation_id,
    )
    assert first.enqueue_generation == 1
    assert second.enqueue_generation == 2


def _mark_enqueued(
    repo: ManagedWorkspaceRepository,
    *,
    operation: WorkspaceOperation,
    generation: int,
    task_id: str,
) -> None:
    from local_workspace_application.workspaces.connected_source_models import (
        ConnectedSourceSyncEnqueueIntent,
    )

    repo.put_connected_source_sync_enqueue_intent(
        ConnectedSourceSyncEnqueueIntent(
            tenant_id=operation.tenant_id,
            workspace_id=operation.workspace_id,
            source_id=operation.source_id,
            indexed_source_binding_id=_INDEXED,
            knowledge_source_binding_ref=_BINDING,
            operation_id=operation.operation_id,
            enqueue_generation=generation,
            last_enqueued_generation=generation,
            last_task_id=task_id,
            last_queue_provider="document_store",
            updated_at=_NOW,
            ownership_classification="COMPLETE_OWNERSHIP",
        )
    )


def test_pending_task_is_reused_without_new_enqueue() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    queue = DocumentStoreTaskQueue(store)
    wiring_context = ToolWiringContext(message_bus=queue)
    operation = _queued_operation()
    repo.put_operation(operation)
    handle = queue.enqueue(
        TaskRequest(
            tenant_id=_TENANT,
            run_id=operation.operation_id,
            task_name=LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
            payload=encode_managed_workspace_sync_job(
                ManagedWorkspaceSyncJob(
                    tenant_id=_TENANT,
                    workspace_id=_WORKSPACE,
                    source_id=_SOURCE,
                    operation_id=_OPERATION,
                )
            ),
            idempotency_key="pending-reuse",
        )
    )
    _mark_enqueued(repo, operation=operation, generation=1, task_id=handle.task_id)
    from local_workspace_application.workspaces.connected_source_sync_enqueue import (
        try_enqueue_connected_source_sync,
    )

    result = try_enqueue_connected_source_sync(
        repository=repo,
        wiring_context=wiring_context,
        operation=operation,
    )
    assert result.enqueued is False
    assert result.enqueue_generation == 1


def test_running_task_is_reused_without_new_enqueue() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    queue = DocumentStoreTaskQueue(store)
    wiring_context = ToolWiringContext(message_bus=queue)
    operation = _queued_operation()
    repo.put_operation(operation)
    handle = queue.enqueue(
        TaskRequest(
            tenant_id=_TENANT,
            run_id=operation.operation_id,
            task_name=LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
            payload=encode_managed_workspace_sync_job(
                ManagedWorkspaceSyncJob(
                    tenant_id=_TENANT,
                    workspace_id=_WORKSPACE,
                    source_id=_SOURCE,
                    operation_id=_OPERATION,
                )
            ),
            idempotency_key="running-reuse",
        )
    )
    from intergrax.queueing.contracts.task_queue import TaskHandle

    running_handle = TaskHandle(
        task_id=handle.task_id,
        provider=handle.provider,
        tenant_id=handle.tenant_id,
    )
    queue.claim_pending(tenant_id=_TENANT, limit=1)
    _mark_enqueued(repo, operation=operation, generation=1, task_id=running_handle.task_id)
    from local_workspace_application.workspaces.connected_source_sync_enqueue import (
        try_enqueue_connected_source_sync,
    )

    result = try_enqueue_connected_source_sync(
        repository=repo,
        wiring_context=wiring_context,
        operation=operation,
    )
    assert result.enqueued is False


def test_failed_task_allocates_next_generation() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    queue = DocumentStoreTaskQueue(store)
    wiring_context = ToolWiringContext(message_bus=queue)
    operation = _queued_operation()
    repo.put_operation(operation)
    handle = queue.enqueue(
        TaskRequest(
            tenant_id=_TENANT,
            run_id=operation.operation_id,
            task_name=LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
            payload=encode_managed_workspace_sync_job(
                ManagedWorkspaceSyncJob(
                    tenant_id=_TENANT,
                    workspace_id=_WORKSPACE,
                    source_id=_SOURCE,
                    operation_id=_OPERATION,
                )
            ),
            idempotency_key="failed-next",
        )
    )
    from intergrax.queueing.contracts.task_queue import TaskHandle

    task_handle = TaskHandle(
        task_id=handle.task_id,
        provider=handle.provider,
        tenant_id=handle.tenant_id,
    )
    claimed = queue.claim_pending(tenant_id=_TENANT, limit=1)
    assert claimed
    queue.mark_failed(task_handle, error_message="boom")
    _mark_enqueued(repo, operation=operation, generation=1, task_id=handle.task_id)
    from local_workspace_application.workspaces.connected_source_sync_enqueue import (
        try_enqueue_connected_source_sync,
    )

    result = try_enqueue_connected_source_sync(
        repository=repo,
        wiring_context=wiring_context,
        operation=operation,
    )
    assert result.enqueued is True
    assert result.enqueue_generation == 2


def test_succeeded_task_allocates_next_generation() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    queue = DocumentStoreTaskQueue(store)
    wiring_context = ToolWiringContext(message_bus=queue)
    operation = _queued_operation()
    repo.put_operation(operation)
    handle = queue.enqueue(
        TaskRequest(
            tenant_id=_TENANT,
            run_id=operation.operation_id,
            task_name=LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
            payload=encode_managed_workspace_sync_job(
                ManagedWorkspaceSyncJob(
                    tenant_id=_TENANT,
                    workspace_id=_WORKSPACE,
                    source_id=_SOURCE,
                    operation_id=_OPERATION,
                )
            ),
            idempotency_key="succeeded-next",
        )
    )
    from intergrax.queueing.contracts.task_queue import TaskHandle

    task_handle = TaskHandle(
        task_id=handle.task_id,
        provider=handle.provider,
        tenant_id=handle.tenant_id,
    )
    queue.claim_pending(tenant_id=_TENANT, limit=1)
    queue.mark_succeeded(task_handle)
    _mark_enqueued(repo, operation=operation, generation=1, task_id=handle.task_id)
    from local_workspace_application.workspaces.connected_source_sync_enqueue import (
        try_enqueue_connected_source_sync,
    )

    result = try_enqueue_connected_source_sync(
        repository=repo,
        wiring_context=wiring_context,
        operation=operation,
    )
    assert result.enqueued is True
    assert result.enqueue_generation == 2


def test_missing_task_allocates_next_generation() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    queue = DocumentStoreTaskQueue(store)
    wiring_context = ToolWiringContext(message_bus=queue)
    operation = _queued_operation()
    repo.put_operation(operation)
    _mark_enqueued(repo, operation=operation, generation=1, task_id="dstq_missing_task")
    from local_workspace_application.workspaces.connected_source_sync_enqueue import (
        try_enqueue_connected_source_sync,
    )

    result = try_enqueue_connected_source_sync(
        repository=repo,
        wiring_context=wiring_context,
        operation=operation,
    )
    assert result.enqueued is True
    assert result.enqueue_generation == 2


def test_terminal_operation_is_not_enqueued() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    queue = DocumentStoreTaskQueue(store)
    wiring_context = ToolWiringContext(message_bus=queue)
    operation = _queued_operation().model_copy(
        update={
            "status": WorkspaceOperationStatus.COMPLETED,
            "completed_at": _NOW,
        }
    )
    from local_workspace_application.workspaces.connected_source_sync_enqueue import (
        try_enqueue_connected_source_sync,
    )

    result = try_enqueue_connected_source_sync(
        repository=repo,
        wiring_context=wiring_context,
        operation=operation,
    )
    assert result.enqueued is False
    assert result.error == "operation_terminal"


def test_completed_operation_syncing_source_projection_repair() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    repo.put_operation(
        _queued_operation().model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "completed_at": _NOW,
                "created_at": _NOW,
            }
        )
    )
    repo.put_source(
        repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE).model_copy(  # type: ignore[union-attr]
            update={"status": WorkspaceSourceStatus.SYNCING}
        )
    )
    from local_workspace_application.workspaces.connected_source_source_projection import (
        repair_connected_source_source_projection,
    )

    repaired = repair_connected_source_source_projection(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
    )
    assert repaired is not None
    assert repaired.status is WorkspaceSourceStatus.READY
    assert repaired.last_sync_at == _NOW


def test_failed_operation_syncing_source_projection_repair() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    repo.put_operation(
        _queued_operation().model_copy(
            update={
                "status": WorkspaceOperationStatus.FAILED,
                "completed_at": _NOW,
                "created_at": _NOW,
            }
        )
    )
    repo.put_source(
        repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE).model_copy(  # type: ignore[union-attr]
            update={"status": WorkspaceSourceStatus.SYNCING}
        )
    )
    from local_workspace_application.workspaces.connected_source_source_projection import (
        repair_connected_source_source_projection,
    )

    repaired = repair_connected_source_source_projection(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
    )
    assert repaired is not None
    assert repaired.status is WorkspaceSourceStatus.ERROR


def test_older_completed_newer_queued_source_projection_syncing() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    repo.put_operation(
        _queued_operation(operation_id="op-old").model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "completed_at": _NOW,
                "created_at": _NOW,
            }
        )
    )
    repo.put_operation(
        _queued_operation(operation_id="op-new").model_copy(
            update={
                "status": WorkspaceOperationStatus.QUEUED,
                "created_at": datetime(2024, 6, 2, 12, 0, tzinfo=UTC),
            }
        )
    )
    from local_workspace_application.workspaces.connected_source_source_projection import (
        project_connected_source_source_status,
    )

    status = project_connected_source_source_status(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
    )
    assert status is WorkspaceSourceStatus.SYNCING


def test_source_projection_repair_is_idempotent() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    repo.put_operation(
        _queued_operation().model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "completed_at": _NOW,
                "created_at": _NOW,
            }
        )
    )
    repo.put_source(
        repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE).model_copy(  # type: ignore[union-attr]
            update={"status": WorkspaceSourceStatus.SYNCING}
        )
    )
    from local_workspace_application.workspaces.connected_source_source_projection import (
        repair_connected_source_source_projection,
    )

    first = repair_connected_source_source_projection(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
    )
    second = repair_connected_source_source_projection(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
    )
    assert first is not None
    assert second is not None
    assert first.status == second.status
    assert first.last_sync_at == second.last_sync_at


def test_accounting_cas_retry_reloads_aggregate_after_concurrent_insert() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_operation(_queued_operation())
    delivery_b = "b" * 64
    repo.put_connected_source_delivery_receipt(_completed_receipt())
    repo.put_connected_source_delivery_receipt(
        _completed_receipt().model_copy(
            update={"delivery_id": delivery_b, "documents_indexed": 2, "documents_unchanged": 1}
        )
    )
    from local_workspace_application.workspaces.connected_source_models import (
        ConnectedSourceOperationDeliveryAccounting,
    )
    from local_workspace_application.workspaces.connected_source_operation_accounting import (
        apply_completed_delivery_accounting,
    )

    operation = repo.get_operation(tenant_id=_TENANT, operation_id=_OPERATION)
    assert operation is not None

    original_replace = repo.replace_operation_if_match
    cas_attempts = 0

    def _replace_with_race(*, expected, replacement):
        nonlocal cas_attempts
        cas_attempts += 1
        if cas_attempts == 1:
            repo.put_connected_source_delivery_accounting_if_absent(
                ConnectedSourceOperationDeliveryAccounting(
                    tenant_id=_TENANT,
                    operation_id=_OPERATION,
                    delivery_id=delivery_b,
                    documents_indexed=2,
                    documents_unchanged=1,
                    items_failed=0,
                    accounted_at=_NOW,
                )
            )
            return False
        return original_replace(expected=expected, replacement=replacement)

    repo.replace_operation_if_match = _replace_with_race  # type: ignore[method-assign]

    updated, result = apply_completed_delivery_accounting(
        repository=repo,
        operation=operation,
        delivery_id=_DELIVERY,
    )
    assert result.applied is True
    assert cas_attempts >= 2
    assert updated.documents_indexed == 3
    assert updated.documents_unchanged == 1


def _seed_many_queue_tasks(
    queue: DocumentStoreTaskQueue,
    *,
    count: int,
) -> None:
    for index in range(count):
        queue.enqueue(
            TaskRequest(
                tenant_id=_TENANT,
                run_id=f"older-{index}",
                task_name="older-task",
                payload=b"older",
                idempotency_key=f"older-task-{index}",
            )
        )
        claimed = queue.claim_pending(tenant_id=_TENANT, limit=1)
        assert claimed
        queue.mark_succeeded(claimed[0][0])


def test_pending_task_beyond_list_limit_is_reused_without_new_enqueue() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    queue = DocumentStoreTaskQueue(store)
    wiring_context = ToolWiringContext(message_bus=queue)
    operation = _queued_operation()
    repo.put_operation(operation)
    _seed_many_queue_tasks(queue, count=501)
    pending_handle = queue.enqueue(
        TaskRequest(
            tenant_id=_TENANT,
            run_id=operation.operation_id,
            task_name=LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
            payload=encode_managed_workspace_sync_job(
                ManagedWorkspaceSyncJob(
                    tenant_id=_TENANT,
                    workspace_id=_WORKSPACE,
                    source_id=_SOURCE,
                    operation_id=_OPERATION,
                )
            ),
            idempotency_key="pending-beyond-limit",
        )
    )
    _mark_enqueued(repo, operation=operation, generation=1, task_id=pending_handle.task_id)
    from local_workspace_application.workspaces.connected_source_sync_enqueue import (
        try_enqueue_connected_source_sync,
    )

    result = try_enqueue_connected_source_sync(
        repository=repo,
        wiring_context=wiring_context,
        operation=operation,
    )
    assert result.enqueued is False
    assert result.enqueue_generation == 1


def test_running_task_beyond_list_limit_is_reused_without_new_enqueue() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    queue = DocumentStoreTaskQueue(store)
    wiring_context = ToolWiringContext(message_bus=queue)
    operation = _queued_operation()
    repo.put_operation(operation)
    _seed_many_queue_tasks(queue, count=501)
    running_handle = queue.enqueue(
        TaskRequest(
            tenant_id=_TENANT,
            run_id=operation.operation_id,
            task_name=LKW_MANAGED_WORKSPACE_SYNC_TASK_NAME,
            payload=encode_managed_workspace_sync_job(
                ManagedWorkspaceSyncJob(
                    tenant_id=_TENANT,
                    workspace_id=_WORKSPACE,
                    source_id=_SOURCE,
                    operation_id=_OPERATION,
                )
            ),
            idempotency_key="running-beyond-limit",
        )
    )
    queue.claim_pending(tenant_id=_TENANT, limit=1)
    _mark_enqueued(repo, operation=operation, generation=1, task_id=running_handle.task_id)
    from local_workspace_application.workspaces.connected_source_sync_enqueue import (
        try_enqueue_connected_source_sync,
    )

    result = try_enqueue_connected_source_sync(
        repository=repo,
        wiring_context=wiring_context,
        operation=operation,
    )
    assert result.enqueued is False
    assert result.enqueue_generation == 1


def test_older_completed_newer_failed_source_projection_error() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    repo.put_operation(
        _queued_operation(operation_id="op-completed").model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "completed_at": _NOW,
                "created_at": _NOW,
            }
        )
    )
    repo.put_operation(
        _queued_operation(operation_id="op-failed").model_copy(
            update={
                "status": WorkspaceOperationStatus.FAILED,
                "completed_at": datetime(2024, 6, 2, 12, 0, tzinfo=UTC),
                "created_at": datetime(2024, 6, 2, 12, 0, tzinfo=UTC),
            }
        )
    )
    from local_workspace_application.workspaces.connected_source_source_projection import (
        project_connected_source_source_status,
    )

    status = project_connected_source_source_status(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
    )
    assert status is WorkspaceSourceStatus.ERROR


def test_older_failed_newer_completed_source_projection_ready() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    repo.put_operation(
        _queued_operation(operation_id="op-failed").model_copy(
            update={
                "status": WorkspaceOperationStatus.FAILED,
                "completed_at": _NOW,
                "created_at": _NOW,
            }
        )
    )
    repo.put_operation(
        _queued_operation(operation_id="op-completed").model_copy(
            update={
                "status": WorkspaceOperationStatus.COMPLETED,
                "completed_at": datetime(2024, 6, 2, 12, 0, tzinfo=UTC),
                "created_at": datetime(2024, 6, 2, 12, 0, tzinfo=UTC),
            }
        )
    )
    from local_workspace_application.workspaces.connected_source_source_projection import (
        project_connected_source_source_status,
    )

    status = project_connected_source_source_status(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
    )
    assert status is WorkspaceSourceStatus.READY


def test_source_projection_handles_missing_operation_timestamps() -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-no-ts",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.COMPLETED,
            completed_at=_NOW,
        )
    )
    from local_workspace_application.workspaces.connected_source_source_projection import (
        project_connected_source_source_status,
        repair_connected_source_source_projection,
    )

    status = project_connected_source_source_status(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
    )
    assert status is WorkspaceSourceStatus.READY
    repaired = repair_connected_source_source_projection(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
    )
    assert repaired is not None
    assert repaired.status is WorkspaceSourceStatus.READY


@pytest.mark.asyncio
async def test_indexed_binding_not_found_repairs_source_to_error(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
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
    repo.put_operation(_queued_operation())
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
            WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE: (
                DisableIndexedSourceMutationHandler()
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
    connected_sync = wiring.connected_source_sync_service

    result = await connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=_OPERATION,
    )

    assert result.status is WorkspaceOperationStatus.FAILED
    assert result.error == "indexed_source_binding_not_found"
    source = repo.get_source(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
    )
    assert source is not None
    assert source.status is WorkspaceSourceStatus.ERROR
