# © Artur Czarnecki. All rights reserved.

"""Tests for connected source sync sink delivery receipts and validation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from local_workspace_application.workspaces.connected_source_delivery import (
    ConnectedSourceDeliveryStatus,
)
from local_workspace_application.workspaces.connected_source_ids import (
    connected_source_id,
    indexed_source_binding_id,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceSyncSinkError,
)
from local_workspace_application.workspaces.connected_source_sync_sink import (
    ConnectedSourceSyncSinkContext,
    WorkspaceConnectedSourceKnowledgeSyncSink,
)
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingResult,
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
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.materialization_visibility import (
    RepositoryKnowledgeMaterializationVisibility,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceDocumentReference,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
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

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_BINDING = "ksb-test"
_SOURCE = connected_source_id(_TENANT, _WORKSPACE, _BINDING)
_INDEXED = indexed_source_binding_id(_TENANT, _WORKSPACE, _BINDING)
_OPERATION = "op-test"
_DELIVERY = "a" * 64
_DELIVERY_2 = "b" * 64
_DELIVERY_3 = "c" * 64


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


def _indexed_binding(
    *,
    status: WorkspaceIndexedSourceBindingStatusV1,
    effective_revision: int,
    mutation_id: str,
) -> WorkspaceIndexedSourceBinding:
    return WorkspaceIndexedSourceBinding(
        indexed_source_binding_id=_INDEXED,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_source_binding_ref=_BINDING,
        source_id=_SOURCE,
        sync_mode=IndexedSourceSyncModeV1.FULL,
        status=status,
        audience_eligibility=IndexedSourceAudienceEligibilityV1.PERSONAL_ONLY,
        mutation_id=mutation_id,
        effective_revision=effective_revision,
        semantic_identity_hash="a" * 64,
        created_at=_NOW,
        updated_at=_NOW,
        cached_safe_display_label="#project-orion",
    )


def _creation_mutation(**overrides: object) -> WorkspaceKnowledgeMutationRecord:
    payload = {
        "mutation_id": "mut-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "operation": WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
        "idempotency_key_hash": "f" * 64,
        "normalized_request_hash": "a" * 64,
        "semantic_identity_hash": "a" * 64,
        "target_revision": 1,
        "committed_revision": 1,
        "status": WorkspaceKnowledgeMutationStatusV1.COMMITTED,
        "outcome": WorkspaceKnowledgeMutationOutcomeV1.APPLIED,
        "result_entity_type": "indexed_source_binding",
        "result_entity_id": _INDEXED,
        "created_at": _NOW,
        "updated_at": _NOW,
        "committed_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceKnowledgeMutationRecord(**payload)


def _bump_head(repo: ManagedWorkspaceRepository, *, committed_revision: int) -> None:
    head_mod = __import__(
        "local_workspace_application.workspaces.knowledge_configuration_models",
        fromlist=["WorkspaceKnowledgeConfigurationHead"],
    )
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    repo.replace_knowledge_configuration_head_if_match(
        expected=head,
        replacement=head_mod.WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=committed_revision,
            updated_at=_NOW,
        ),
    )


def _batch(
    *,
    envelopes: tuple[KnowledgeSyncEnvelope, ...] = (),
    binding_configuration_version: int = 1,
    delivery_id: str = _DELIVERY,
) -> KnowledgeSyncBatch:
    return KnowledgeSyncBatch(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        binding_configuration_version=binding_configuration_version,
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
        delivery_id=delivery_id,
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
    configuration_version = 1

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
            configuration_version=self.configuration_version,
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
    repo.put_knowledge_configuration_mutation_if_absent(_creation_mutation())
    tenant_binding_port = _TenantBindingPort()
    binding_repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    binding_repo.create(
        tenant_binding_port.get_binding(tenant_id=_TENANT, binding_id=_BINDING)
    )
    config = WorkspaceKnowledgeConfigurationService(repo, _Lookup())
    indexing = AsyncMock()
    document_number = 0

    async def _index(**kwargs):
        nonlocal document_number
        document_number += 1
        document_id = f"doc-{document_number}"
        ownership = kwargs["materialization_ownership"]
        repo.put_document_ref(
            WorkspaceDocumentReference(
                document_id=document_id,
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                source_id=_SOURCE,
                source_path=f"connected/{ownership.remote_id}.md",
                file_name=kwargs["safe_file_name"],
                content_hash=kwargs["content_hash"],
                indexed_at=_NOW,
                materialization_ownership=ownership,
                visibility_authority_ref=ownership.delivery_id,
                visibility_authority_type="delivery_receipt",
            )
        )
        return WorkspaceDocumentIndexingResult(
            indexed=True,
            unchanged=False,
            document_id=document_id,
            documents_indexed=1,
        )
    indexing.index_connected_source_one = AsyncMock(side_effect=_index)
    sink = WorkspaceConnectedSourceKnowledgeSyncSink(
        repository=repo,
        indexing_service=indexing,
        configuration_reader=config,
        tenant_binding_port=tenant_binding_port,
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
    _repo, sink, indexing = sink_env
    batch = _batch(envelopes=(_envelope(KnowledgeChangeKind.UPSERT),))
    first = await sink.apply_batch(batch=batch)
    second = await sink.apply_batch(batch=batch)
    assert first.replayed is False
    assert second.replayed is True
    assert indexing.index_connected_source_one.await_count == 1


@pytest.mark.asyncio
async def test_materialization_supersession_uses_delivery_sequence_and_replay_is_stale(
    sink_env,
) -> None:
    repo, sink, indexing = sink_env
    resolver = RepositoryKnowledgeMaterializationVisibility(repo)

    await sink.apply_batch(
        batch=_batch(
            envelopes=(_envelope(KnowledgeChangeKind.UPSERT, text="v1"),),
            delivery_id=_DELIVERY,
            binding_configuration_version=1,
        )
    )
    refs = repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    first = next(ref for ref in refs if ref.materialization_ownership is not None)
    assert first.materialization_ownership is not None
    assert resolver.is_visible(ownership=first.materialization_ownership)

    await sink.apply_batch(
        batch=_batch(
            envelopes=(_envelope(KnowledgeChangeKind.UPSERT, text="v2"),),
            delivery_id=_DELIVERY_2,
            binding_configuration_version=1,
        )
    )
    first_receipt = repo.get_connected_source_delivery_receipt(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        delivery_id=_DELIVERY,
    )
    second_receipt = repo.get_connected_source_delivery_receipt(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        delivery_id=_DELIVERY_2,
    )
    assert first_receipt is not None
    assert second_receipt is not None
    assert first_receipt.binding_configuration_version == 1
    assert second_receipt.binding_configuration_version == 1
    assert first_receipt.materialization_sequence == 1
    assert second_receipt.materialization_sequence == 2
    refs = repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    second = next(
        ref
        for ref in refs
        if ref.materialization_ownership is not None
        and ref.materialization_ownership.delivery_id == _DELIVERY_2
    )
    assert second.materialization_ownership is not None
    assert not resolver.is_visible(ownership=first.materialization_ownership)
    assert resolver.is_visible(ownership=second.materialization_ownership)

    await sink.apply_batch(
        batch=_batch(
            envelopes=(_envelope(KnowledgeChangeKind.UPSERT, text="v1"),),
            delivery_id=_DELIVERY,
            binding_configuration_version=1,
        )
    )
    assert resolver.is_visible(ownership=second.materialization_ownership)

    replay = await sink.apply_batch(
        batch=_batch(
            envelopes=(_envelope(KnowledgeChangeKind.UPSERT, text="v2"),),
            delivery_id=_DELIVERY_2,
            binding_configuration_version=2,
        )
    )
    assert replay.replayed is True
    assert resolver.is_visible(ownership=second.materialization_ownership)
    assert indexing.index_connected_source_one.await_count == 2


@pytest.mark.asyncio
async def test_configuration_change_keeps_delivery_sequence_order(
    sink_env,
    monkeypatch,
) -> None:
    repo, sink, _indexing = sink_env
    await sink.apply_batch(
        batch=_batch(
            envelopes=(_envelope(KnowledgeChangeKind.UPSERT, text="v1"),),
            delivery_id=_DELIVERY,
            binding_configuration_version=1,
        )
    )
    first_ref = next(
        ref
        for ref in repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)
        if ref.materialization_ownership is not None
    )
    assert first_ref.materialization_ownership is not None
    remote_id = first_ref.materialization_ownership.remote_id
    assert remote_id is not None
    sink._tenant_binding_port.configuration_version = 2
    original_replace = repo.replace_active_materialization_pointer
    raced = False

    def _replace_with_concurrent_winner(*, expected, replacement):
        nonlocal raced
        if not raced:
            raced = True
            assert original_replace(expected=expected, replacement=replacement)
            return False
        return original_replace(expected=expected, replacement=replacement)

    monkeypatch.setattr(
        repo,
        "replace_active_materialization_pointer",
        _replace_with_concurrent_winner,
    )
    await sink.apply_batch(
        batch=_batch(
            envelopes=(_envelope(KnowledgeChangeKind.UPSERT, text="v2"),),
            delivery_id=_DELIVERY_2,
            binding_configuration_version=2,
        )
    )
    pointer = repo.get_active_materialization_pointer(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        remote_id=remote_id,
    )
    assert pointer is not None
    assert pointer.materialization_revision == 2
    assert pointer.delivery_id == _DELIVERY_2
    receipt = repo.get_connected_source_delivery_receipt(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        delivery_id=_DELIVERY_2,
    )
    assert receipt is not None
    assert receipt.binding_configuration_version == 2
    assert receipt.materialization_sequence == 2
    assert raced


@pytest.mark.asyncio
async def test_equal_configuration_version_does_not_conflict(sink_env) -> None:
    repo, sink, _indexing = sink_env
    await sink.apply_batch(
        batch=_batch(
            envelopes=(_envelope(KnowledgeChangeKind.UPSERT),),
            delivery_id=_DELIVERY_2,
            binding_configuration_version=1,
        )
    )
    await sink.apply_batch(
        batch=_batch(
            envelopes=(_envelope(KnowledgeChangeKind.UPSERT),),
            delivery_id=_DELIVERY_3,
            binding_configuration_version=1,
        )
    )
    receipt = repo.get_connected_source_delivery_receipt(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        delivery_id=_DELIVERY_3,
    )
    assert receipt is not None
    assert receipt.materialization_sequence == 2


@pytest.mark.asyncio
async def test_temp_file_cleaned_up_after_success(sink_env, monkeypatch) -> None:
    _repo, sink, _indexing = sink_env
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
    indexing.index_connected_source_one = AsyncMock(side_effect=__import__(
        "local_workspace_application.workspaces.document_indexing",
        fromlist=["WorkspaceDocumentIndexingError"],
    ).WorkspaceDocumentIndexingError("failed"))
    with pytest.raises(ConnectedSourceSyncSinkError):
        await sink.apply_batch(batch=_batch(envelopes=(_envelope(KnowledgeChangeKind.UPSERT),)))
    assert created
    assert not created[0].exists()


@pytest.mark.asyncio
async def test_reactivated_binding_batch_validation_succeeds(sink_env) -> None:
    repo, sink, indexing = sink_env
    repo.put_knowledge_configuration_mutation_if_absent(_creation_mutation())
    repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_binding(
            status=WorkspaceIndexedSourceBindingStatusV1.DISABLED,
            effective_revision=2,
            mutation_id="mut-2",
        )
    )
    repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_binding(
            status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
            effective_revision=3,
            mutation_id="mut-3",
        )
    )
    _bump_head(repo, committed_revision=3)
    await sink.apply_batch(batch=_batch(envelopes=(_envelope(KnowledgeChangeKind.UPSERT),)))
    assert indexing.index_connected_source_one.await_count == 1


@pytest.mark.asyncio
async def test_disabled_binding_rejected(sink_env) -> None:
    repo, sink, _ = sink_env
    repo.put_knowledge_configuration_mutation_if_absent(_creation_mutation())
    repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_binding(
            status=WorkspaceIndexedSourceBindingStatusV1.DISABLED,
            effective_revision=2,
            mutation_id="mut-2",
        )
    )
    _bump_head(repo, committed_revision=2)
    with pytest.raises(ConnectedSourceSyncSinkError, match="connected_source_indexed_binding_inactive"):
        await sink.apply_batch(batch=_batch())


@pytest.mark.asyncio
async def test_corrupt_source_origin_rejected(sink_env) -> None:
    repo, sink, _ = sink_env
    repo.put_knowledge_configuration_mutation_if_absent(_creation_mutation())
    source = repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE)
    assert source is not None
    repo.put_source(
        WorkspaceSource(
            source_id=source.source_id,
            workspace_id=source.workspace_id,
            tenant_id=source.tenant_id,
            source_type=source.source_type,
            path=source.path,
            recursive=source.recursive,
            status=source.status,
            created_at=source.created_at,
            knowledge_configuration_creation_mutation_id="mut-corrupt",
            knowledge_configuration_visibility_revision=source.knowledge_configuration_visibility_revision,
        )
    )
    with pytest.raises(ConnectedSourceSyncSinkError, match="connected_source_workspace_source_uncommitted"):
        await sink.apply_batch(batch=_batch())
