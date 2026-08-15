# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceOperationDeliveryAccounting,
    ConnectedSourceSyncEnqueueIntent,
)
from local_workspace_application.workspaces.connected_source_recovery_ownership_index import (
    ConnectedSourceRecoveryOwnershipIndexError,
    RecoveryRecordKindV1,
    canonical_record_fingerprint,
    index_entry_for_delivery_accounting,
    index_entry_for_enqueue_intent,
    index_entry_for_index_receipt,
    recovery_ownership_index_partition,
)
from local_workspace_application.workspaces.document_indexing import (
    _index_receipt_partition,
    _index_receipt_row_key,
    _WorkspaceDocumentIndexReceipt,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationOwnershipV1,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import DocumentRecord

_NOW = datetime(2026, 8, 6, tzinfo=UTC)
_DELIVERY = "a" * 64


def _ownership(
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "workspace-a",
    source_id: str = "source-a",
    binding_id: str = "binding-a",
    binding_ref: str = "knowledge-binding-a",
) -> KnowledgeMaterializationOwnershipV1:
    return KnowledgeMaterializationOwnershipV1.connected(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        indexed_source_binding_id=binding_id,
        knowledge_source_binding_ref=binding_ref,
        delivery_id=_DELIVERY,
        remote_id="remote-a",
        materialization_generation="generation-1",
        materialization_sequence=1,
    )


def _put_receipt(
    repository: ManagedWorkspaceRepository,
    *,
    ownership: KnowledgeMaterializationOwnershipV1,
    operation_id: str,
    document_id: str,
) -> _WorkspaceDocumentIndexReceipt:
    receipt = _WorkspaceDocumentIndexReceipt(
        tenant_id=ownership.tenant_id,
        workspace_id=ownership.workspace_id,
        source_id=ownership.source_id,
        operation_id=operation_id,
        logical_source_path=f"/{document_id}.md",
        safe_file_name=f"{document_id}.md",
        content_hash=f"hash-{document_id}",
        document_id=document_id,
        status="completed",
        created_at=_NOW,
        completed_at=_NOW,
        materialization_scope=ownership.identity_scope,
        materialization_ownership=ownership,
    )
    partition = _index_receipt_partition(ownership.tenant_id)
    row_key = _index_receipt_row_key(
        tenant_id=ownership.tenant_id,
        workspace_id=ownership.workspace_id,
        source_id=ownership.source_id,
        logical_source_path=receipt.logical_source_path,
        content_hash=receipt.content_hash,
        materialization_scope=receipt.materialization_scope,
    )
    assert repository.document_store.put_if_absent(
        DocumentRecord(
            partition_key=partition,
            row_key=row_key,
            data=receipt.model_dump(mode="json"),
        )
    )
    repository.put_connected_source_recovery_ownership_index_entry(
        index_entry_for_index_receipt(
            receipt,
            canonical_partition_key=partition,
            canonical_row_key=row_key,
        )
    )
    return receipt


def _seed_all_kinds(
    repository: ManagedWorkspaceRepository,
    *,
    ownership: KnowledgeMaterializationOwnershipV1,
    operation_id: str,
    document_id: str,
) -> None:
    _put_receipt(
        repository,
        ownership=ownership,
        operation_id=operation_id,
        document_id=document_id,
    )
    repository.put_connected_source_sync_enqueue_intent(
        ConnectedSourceSyncEnqueueIntent(
            tenant_id=ownership.tenant_id,
            workspace_id=ownership.workspace_id,
            source_id=ownership.source_id,
            indexed_source_binding_id=ownership.indexed_source_binding_id,
            knowledge_source_binding_ref=ownership.knowledge_source_binding_ref,
            operation_id=operation_id,
            enqueue_generation=1,
            updated_at=_NOW,
            ownership_classification="COMPLETE_OWNERSHIP",
        )
    )
    repository.put_connected_source_delivery_accounting_if_absent(
        ConnectedSourceOperationDeliveryAccounting(
            tenant_id=ownership.tenant_id,
            workspace_id=ownership.workspace_id,
            source_id=ownership.source_id,
            indexed_source_binding_id=ownership.indexed_source_binding_id,
            knowledge_source_binding_ref=ownership.knowledge_source_binding_ref,
            operation_id=operation_id,
            delivery_id=_DELIVERY,
            documents_indexed=1,
            documents_unchanged=0,
            items_failed=0,
            accounted_at=_NOW,
            ownership_classification="COMPLETE_OWNERSHIP",
        )
    )


@pytest.mark.parametrize(
    "record_kind",
    [
        RecoveryRecordKindV1.INDEX_RECEIPT,
        RecoveryRecordKindV1.ENQUEUE_INTENT,
        RecoveryRecordKindV1.DELIVERY_ACCOUNTING,
    ],
)
def test_exact_owner_query_isolates_binding_workspace_and_tenant(
    record_kind: RecoveryRecordKindV1,
) -> None:
    store = InMemoryDocumentStore(cursor_secret=b"recovery-index-secret")
    repository = ManagedWorkspaceRepository(store)
    binding_a = _ownership()
    binding_b = _ownership(
        binding_id="binding-b",
        binding_ref="knowledge-binding-b",
    )
    other_workspace = _ownership(workspace_id="workspace-b")
    other_tenant = _ownership(tenant_id="tenant-b")
    _seed_all_kinds(
        repository,
        ownership=binding_a,
        operation_id="operation-a",
        document_id="document-a",
    )
    _seed_all_kinds(
        repository,
        ownership=binding_b,
        operation_id="operation-b",
        document_id="document-b",
    )
    _seed_all_kinds(
        repository,
        ownership=other_workspace,
        operation_id="operation-ws",
        document_id="document-ws",
    )
    _seed_all_kinds(
        repository,
        ownership=other_tenant,
        operation_id="operation-tenant",
        document_id="document-tenant",
    )

    page = repository.list_connected_source_recovery_records_by_owner(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        record_kind=record_kind,
        limit=10,
    )
    assert len(page.index_entries) == 1
    assert page.orphan_index_entries == ()
    assert page.index_entries[0].indexed_source_binding_id == "binding-a"
    assert page.index_entries[0].operation_id == "operation-a"

    if record_kind is RecoveryRecordKindV1.INDEX_RECEIPT:
        # Index-receipt row keys include materialization scope, so the same
        # operation id may legally exist under a second binding.
        _put_receipt(
            repository,
            ownership=binding_b,
            operation_id="operation-a",
            document_id="document-shared-op",
        )
        page_b = repository.list_connected_source_recovery_records_by_owner(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            source_id="source-a",
            indexed_source_binding_id="binding-b",
            knowledge_source_binding_ref="knowledge-binding-b",
            record_kind=record_kind,
            limit=10,
        )
        assert {item.operation_id for item in page_b.index_entries} >= {"operation-a"}
        page_a = repository.list_connected_source_recovery_records_by_owner(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            source_id="source-a",
            indexed_source_binding_id="binding-a",
            knowledge_source_binding_ref="knowledge-binding-a",
            record_kind=record_kind,
            limit=10,
        )
        assert all(
            item.indexed_source_binding_id == "binding-a"
            for item in page_a.index_entries
        )


def test_recovery_index_pagination_and_cursor_restart() -> None:
    store = InMemoryDocumentStore(cursor_secret=b"recovery-page-secret")
    repository = ManagedWorkspaceRepository(store)
    ownership = _ownership()
    for index in range(5):
        repository.put_connected_source_sync_enqueue_intent(
            ConnectedSourceSyncEnqueueIntent(
                tenant_id=ownership.tenant_id,
                workspace_id=ownership.workspace_id,
                source_id=ownership.source_id,
                indexed_source_binding_id=ownership.indexed_source_binding_id,
                knowledge_source_binding_ref=ownership.knowledge_source_binding_ref,
                operation_id=f"operation-{index}",
                enqueue_generation=1,
                updated_at=_NOW,
                ownership_classification="COMPLETE_OWNERSHIP",
            )
        )
    first = repository.list_connected_source_recovery_records_by_owner(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        record_kind=RecoveryRecordKindV1.ENQUEUE_INTENT,
        limit=2,
    )
    assert len(first.index_entries) == 2
    assert first.next_cursor is not None
    second = repository.list_connected_source_recovery_records_by_owner(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        record_kind=RecoveryRecordKindV1.ENQUEUE_INTENT,
        limit=2,
        cursor=first.next_cursor,
    )
    assert len(second.index_entries) == 2
    restarted = repository.list_connected_source_recovery_records_by_owner(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        record_kind=RecoveryRecordKindV1.ENQUEUE_INTENT,
        limit=2,
        cursor=first.next_cursor,
    )
    assert [item.operation_id for item in restarted.index_entries] == [
        item.operation_id for item in second.index_entries
    ]


def test_forged_cursor_rejected() -> None:
    store = InMemoryDocumentStore(cursor_secret=b"recovery-forged-secret")
    repository = ManagedWorkspaceRepository(store)
    with pytest.raises(ValueError, match="document_store_cursor"):
        repository.list_connected_source_recovery_records_by_owner(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            source_id="source-a",
            indexed_source_binding_id="binding-a",
            knowledge_source_binding_ref="knowledge-binding-a",
            record_kind=RecoveryRecordKindV1.ENQUEUE_INTENT,
            limit=1,
            cursor="forged-cursor-value",
        )


def test_fingerprint_and_ownership_mismatch_fail_closed() -> None:
    store = InMemoryDocumentStore(cursor_secret=b"recovery-mismatch-secret")
    repository = ManagedWorkspaceRepository(store)
    ownership = _ownership()
    intent = ConnectedSourceSyncEnqueueIntent(
        tenant_id=ownership.tenant_id,
        workspace_id=ownership.workspace_id,
        source_id=ownership.source_id,
        indexed_source_binding_id=ownership.indexed_source_binding_id,
        knowledge_source_binding_ref=ownership.knowledge_source_binding_ref,
        operation_id="operation-mismatch",
        enqueue_generation=1,
        updated_at=_NOW,
        ownership_classification="COMPLETE_OWNERSHIP",
    )
    repository.put_connected_source_sync_enqueue_intent(intent)
    page = repository.list_connected_source_recovery_records_by_owner(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        record_kind=RecoveryRecordKindV1.ENQUEUE_INTENT,
        limit=1,
    )
    entry = page.index_entries[0]
    mutated = intent.model_copy(update={"enqueue_generation": 2, "updated_at": _NOW})
    repository.document_store.put(
        DocumentRecord(
            partition_key=entry.canonical_partition_key,
            row_key=entry.canonical_row_key,
            data=mutated.model_dump(mode="json"),
        )
    )
    with pytest.raises(
        ConnectedSourceRecoveryOwnershipIndexError,
        match="recovery_ownership_index_reference_mismatch",
    ):
        repository.list_connected_source_recovery_records_by_owner(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            source_id="source-a",
            indexed_source_binding_id="binding-a",
            knowledge_source_binding_ref="knowledge-binding-a",
            record_kind=RecoveryRecordKindV1.ENQUEUE_INTENT,
            limit=1,
        )


def test_orphan_index_and_idempotent_repair() -> None:
    store = InMemoryDocumentStore(cursor_secret=b"recovery-orphan-secret")
    repository = ManagedWorkspaceRepository(store)
    ownership = _ownership()
    intent = ConnectedSourceSyncEnqueueIntent(
        tenant_id=ownership.tenant_id,
        workspace_id=ownership.workspace_id,
        source_id=ownership.source_id,
        indexed_source_binding_id=ownership.indexed_source_binding_id,
        knowledge_source_binding_ref=ownership.knowledge_source_binding_ref,
        operation_id="operation-orphan",
        enqueue_generation=1,
        updated_at=_NOW,
        ownership_classification="COMPLETE_OWNERSHIP",
    )
    repository.put_connected_source_sync_enqueue_intent(intent)
    page = repository.list_connected_source_recovery_records_by_owner(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        record_kind=RecoveryRecordKindV1.ENQUEUE_INTENT,
        limit=1,
    )
    entry = page.index_entries[0]
    repository.document_store.delete(
        entry.canonical_partition_key,
        entry.canonical_row_key,
    )
    orphan_page = repository.list_connected_source_recovery_records_by_owner(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        record_kind=RecoveryRecordKindV1.ENQUEUE_INTENT,
        limit=1,
    )
    assert orphan_page.index_entries == ()
    assert orphan_page.orphan_index_entries == (entry,)

    repository.document_store.put(
        DocumentRecord(
            partition_key=entry.canonical_partition_key,
            row_key=entry.canonical_row_key,
            data=intent.model_dump(mode="json"),
        )
    )
    repaired = repository.repair_connected_source_recovery_ownership_index_entry(
        index_entry_for_enqueue_intent(
            intent,
            canonical_partition_key=entry.canonical_partition_key,
            canonical_row_key=entry.canonical_row_key,
        )
    )
    again = repository.repair_connected_source_recovery_ownership_index_entry(repaired)
    assert again == repaired
    assert canonical_record_fingerprint(intent) == repaired.canonical_fingerprint


def test_legacy_records_are_not_indexed() -> None:
    store = InMemoryDocumentStore(cursor_secret=b"recovery-legacy-secret")
    repository = ManagedWorkspaceRepository(store)
    repository.put_connected_source_sync_enqueue_intent(
        ConnectedSourceSyncEnqueueIntent(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            source_id="source-a",
            operation_id="operation-legacy",
            enqueue_generation=1,
            updated_at=_NOW,
            ownership_classification="LEGACY_MIGRATION_REQUIRED",
        )
    )
    page = repository.list_connected_source_recovery_records_by_owner(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        record_kind=RecoveryRecordKindV1.ENQUEUE_INTENT,
        limit=10,
    )
    assert page.index_entries == ()
    assert page.orphan_index_entries == ()
    assert (
        store.query(
            recovery_ownership_index_partition("tenant-a"),
            limit=10,
        ).documents
        == ()
    )


def test_accounting_factory_requires_complete_ownership() -> None:
    accounting = ConnectedSourceOperationDeliveryAccounting.model_construct(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id=None,
        knowledge_source_binding_ref=None,
        operation_id="operation-a",
        delivery_id=_DELIVERY,
        documents_indexed=0,
        documents_unchanged=0,
        items_failed=0,
        accounted_at=_NOW,
        ownership_classification="LEGACY_MIGRATION_REQUIRED",
    )
    with pytest.raises(
        ConnectedSourceRecoveryOwnershipIndexError,
        match="recovery_ownership_index_incomplete",
    ):
        index_entry_for_delivery_accounting(
            accounting,
            canonical_partition_key="p",
            canonical_row_key="r",
        )
