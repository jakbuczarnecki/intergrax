# © Artur Czarnecki. All rights reserved.

"""Tests for connected source sync sink delivery receipts and validation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
)
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeSyncBatch,
    KnowledgeSyncEnvelope,
    KnowledgeSyncMode,
)
from tests.unit.runtime.vendor_knowledge._fakes import make_descriptor
from local_workspace_application.workspaces.connected_source_delivery import (
    ConnectedSourceDeliveryStatus,
)
from local_workspace_application.workspaces.connected_source_models import ConnectedSourceSyncSinkError
from local_workspace_application.workspaces.connected_source_sync_sink import (
    ConnectedSourceSyncSinkContext,
    WorkspaceConnectedSourceKnowledgeSyncSink,
)
from local_workspace_application.workspaces.document_indexing import WorkspaceDocumentIndexingResult
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
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

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_SOURCE = "src:connected:test"
_BINDING = "ksb-test"
_INDEXED = "idx-test"
_OPERATION = "op-test"
_DELIVERY = "a" * 64


def _structured_record(text: str = "marker") -> dict[str, object]:
    return {
        "schema": "slack.conversation.message.knowledge.v1",
        "provider": "slack",
        "source_kind": "slack_conversation",
        "conversation": {"safe_display_name": "#project-orion"},
        "message": {"message_ts": "1704153600.000001"},
        "thread": {"root_thread_ts": None, "reply_count": 0},
        "actor": {"provider_id": "U111"},
        "text": text,
        "timestamps": {"created_at": "2024-01-02T12:00:00+00:00", "edited_at": None},
        "edit_state": {"edited": False},
        "safe_file_inventory": [],
    }


def _batch(*, envelopes: tuple[KnowledgeSyncEnvelope, ...] = ()) -> KnowledgeSyncBatch:
    return KnowledgeSyncBatch(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        binding_configuration_version=1,
        source=KnowledgeSourceRef(
            tenant_id=_TENANT,
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
            source_kind=SLACK_CONVERSATION_SOURCE_KIND,
            connection_ref="conn.slack",
            scope=KnowledgeSourceScope(
                remote_scope_id="scope",
                remote_scope_type="slack_conversation",
                safe_display_name="#project-orion",
                parameters={},
            ),
        ),
        mode=KnowledgeSyncMode.RECONCILIATION,
        delivery_id=_DELIVERY,
        envelopes=envelopes,
        has_more=False,
    )


class _Lookup:
    def require_workspace(self, *, tenant_id: str, workspace_id: str):
        return Workspace(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            name="ws",
            status=WorkspaceStatus.ACTIVE,
            created_at=_NOW,
            updated_at=_NOW,
        )


class _TenantBindingPort:
    def get_binding(self, *, tenant_id: str, binding_id: str):
        return KnowledgeSourceBinding(
            binding_id=binding_id,
            tenant_id=tenant_id,
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
            source_kind=SLACK_CONVERSATION_SOURCE_KIND,
            connection_ref="conn.slack",
            safe_display_name="#project-orion",
            scope=KnowledgeSourceScope(
                remote_scope_id="scope",
                remote_scope_type="slack_conversation",
                safe_display_name="#project-orion",
                parameters={},
            ),
            status=KnowledgeSourceBindingStatus.ACTIVE,
            configuration_version=1,
        )


@pytest.fixture
def sink_env(tmp_path: Path):
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
    binding_repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    binding_repo.create(_TenantBindingPort().get_binding(tenant_id=_TENANT, binding_id=_BINDING))
    config = WorkspaceKnowledgeConfigurationService(repo, _Lookup())
    indexing = AsyncMock()
    indexing.index_one = AsyncMock(
        return_value=WorkspaceDocumentIndexingResult(
            indexed=True,
            unchanged=False,
            document_id="doc-1",
            documents_indexed=1,
        )
    )
    sink = WorkspaceConnectedSourceKnowledgeSyncSink(
        repository=repo,
        indexing_service=indexing,
        configuration_reader=config,
        tenant_binding_port=_TenantBindingPort(),
        context=ConnectedSourceSyncSinkContext(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_INDEXED,
            knowledge_source_binding_ref=_BINDING,
            operation_id=_OPERATION,
        ),
    )
    return repo, sink, indexing


def _envelope(change_kind: KnowledgeChangeKind, *, text: str = "marker") -> KnowledgeSyncEnvelope:
    descriptor = make_descriptor(content_mode=KnowledgeContentMode.STRUCTURED_RECORD)
    if change_kind in {KnowledgeChangeKind.DELETED, KnowledgeChangeKind.REVOKED}:
        return KnowledgeSyncEnvelope(
            remote_id=descriptor.identity.remote_id,
            change_kind=change_kind,
            descriptor=descriptor,
        )
    return KnowledgeSyncEnvelope(
        remote_id=descriptor.identity.remote_id,
        change_kind=change_kind,
        descriptor=descriptor,
        content=KnowledgeContent(
            mode=KnowledgeContentMode.STRUCTURED_RECORD,
            structured_record=_structured_record(text),
            mime_type="application/json",
            content_hash="b" * 64,
        ),
    )


@pytest.mark.asyncio
async def test_empty_batch_completes_receipt(sink_env) -> None:
    repo, sink, _ = sink_env
    await sink.apply_batch(batch=_batch())
    receipt = repo.get_connected_source_delivery_receipt(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        delivery_id=_DELIVERY,
    )
    assert receipt is not None
    assert receipt.status is ConnectedSourceDeliveryStatus.COMPLETED
    assert receipt.completed_at is not None
    assert receipt.items_failed == 0


@pytest.mark.asyncio
async def test_tombstone_rejected(sink_env) -> None:
    _, sink, _ = sink_env
    with pytest.raises(ConnectedSourceSyncSinkError):
        await sink.apply_batch(batch=_batch(envelopes=(_envelope(KnowledgeChangeKind.DELETED),)))


@pytest.mark.asyncio
async def test_permissions_change_rejected(sink_env) -> None:
    _, sink, _ = sink_env
    with pytest.raises(ConnectedSourceSyncSinkError):
        await sink.apply_batch(
            batch=_batch(envelopes=(_envelope(KnowledgeChangeKind.PERMISSIONS_CHANGED),))
        )


@pytest.mark.asyncio
async def test_completed_replay_is_idempotent(sink_env) -> None:
    repo, sink, indexing = sink_env
    batch = _batch(envelopes=(_envelope(KnowledgeChangeKind.UPSERT),))
    first = await sink.apply_batch(batch=batch)
    second = await sink.apply_batch(batch=batch)
    assert first.replayed is False
    assert second.replayed is True
    assert indexing.index_one.await_count == 1


@pytest.mark.asyncio
async def test_temp_file_cleaned_up_after_success(sink_env, monkeypatch) -> None:
    repo, sink, indexing = sink_env
    import tempfile as tempfile_module

    created: list[Path] = []
    real_mkstemp = tempfile_module.mkstemp

    def _mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        created.append(Path(name))
        return fd, name

    monkeypatch.setattr(
        "local_workspace_application.workspaces.connected_source_sync_sink.tempfile.mkstemp",
        _mkstemp,
    )
    await sink.apply_batch(batch=_batch(envelopes=(_envelope(KnowledgeChangeKind.UPSERT),)))
    assert created
    assert not created[0].exists()


@pytest.mark.asyncio
async def test_temp_file_cleaned_up_after_indexing_failure(sink_env, monkeypatch) -> None:
    _, sink, indexing = sink_env
    import tempfile as tempfile_module

    created: list[Path] = []
    real_mkstemp = tempfile_module.mkstemp

    def _mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        created.append(Path(name))
        return fd, name

    monkeypatch.setattr(
        "local_workspace_application.workspaces.connected_source_sync_sink.tempfile.mkstemp",
        _mkstemp,
    )
    indexing.index_one = AsyncMock(side_effect=__import__(
        "local_workspace_application.workspaces.document_indexing",
        fromlist=["WorkspaceDocumentIndexingError"],
    ).WorkspaceDocumentIndexingError("failed"))
    with pytest.raises(ConnectedSourceSyncSinkError):
        await sink.apply_batch(batch=_batch(envelopes=(_envelope(KnowledgeChangeKind.UPSERT),)))
    assert created
    assert not created[0].exists()
