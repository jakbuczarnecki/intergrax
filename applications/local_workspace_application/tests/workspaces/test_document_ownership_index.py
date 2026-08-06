from datetime import UTC, datetime

import pytest

from local_workspace_application.workspaces.document_ownership_index import (
    DocumentOwnershipIndexError,
    WorkspaceDocumentOwnershipIndexEntryV1,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationOwnershipV1,
)
from local_workspace_application.workspaces.models import WorkspaceDocumentReference
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 6, tzinfo=UTC)


def _ownership(
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "workspace-a",
    source_id: str = "source-a",
    binding_id: str = "binding-a",
    binding_ref: str = "knowledge-binding-a",
    remote_id: str = "remote-a",
) -> KnowledgeMaterializationOwnershipV1:
    return KnowledgeMaterializationOwnershipV1.connected(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        indexed_source_binding_id=binding_id,
        knowledge_source_binding_ref=binding_ref,
        delivery_id="d" * 64,
        remote_id=remote_id,
    )


def _reference(
    document_id: str,
    ownership: KnowledgeMaterializationOwnershipV1 | None,
) -> WorkspaceDocumentReference:
    return WorkspaceDocumentReference(
        document_id=document_id,
        tenant_id=ownership.tenant_id if ownership is not None else "tenant-a",
        workspace_id=ownership.workspace_id if ownership is not None else "workspace-a",
        source_id=ownership.source_id if ownership is not None else "local-source",
        source_path=f"/{document_id}.md",
        file_name=f"{document_id}.md",
        content_hash=f"sha256:{document_id:0<64}"[:71],
        indexed_at=_NOW,
        materialization_ownership=ownership,
        visibility_authority_type=(
            "delivery_manifest" if ownership is not None else "legacy_immediate"
        ),
        visibility_authority_ref=(
            ownership.delivery_id if ownership is not None else None
        ),
    )


def test_connected_reference_is_indexed_and_paginated_by_exact_scope() -> None:
    store = InMemoryDocumentStore()
    repository = ManagedWorkspaceRepository(store)
    for index in range(3):
        repository.put_document_ref(
            _reference(
                f"document-{index}",
                _ownership(remote_id=f"remote-{index}"),
            )
        )
    repository.put_document_ref(
        _reference(
            "other-binding",
            _ownership(binding_id="binding-b", binding_ref="knowledge-binding-b"),
        )
    )

    first = repository.list_document_refs_by_materialization_owner(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        limit=2,
        cursor=None,
    )
    assert [item.document_id for item in first.references] == [
        "document-0",
        "document-1",
    ]
    assert first.next_cursor is not None

    second = repository.list_document_refs_by_materialization_owner(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        limit=2,
        cursor=first.next_cursor,
    )
    assert [item.document_id for item in second.references] == ["document-2"]
    assert second.next_cursor is None


def test_index_entry_repair_is_idempotent_and_legacy_refs_are_excluded() -> None:
    store = InMemoryDocumentStore()
    repository = ManagedWorkspaceRepository(store)
    reference = _reference("document-a", _ownership())
    entry = repository.repair_document_ownership_index_entry(reference)
    assert repository.repair_document_ownership_index_entry(reference) == entry

    legacy = _reference("local-document", None)
    repository.put_document_ref(legacy)
    page = repository.list_document_refs_by_materialization_owner(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        limit=10,
        cursor=None,
    )
    assert [item.document_id for item in page.references] == ["document-a"]


def test_index_mismatch_fails_closed_and_missing_reference_is_orphan_evidence() -> None:
    store = InMemoryDocumentStore()
    repository = ManagedWorkspaceRepository(store)
    reference = _reference("document-a", _ownership())
    repository.put_document_ref(reference)
    store.put(
        WorkspaceDocumentOwnershipIndexEntryV1.for_reference(reference).to_document()
    )
    store.delete("lkw.managed_workspace:tenant-a:document", "workspace-a:document-a")
    page = repository.list_document_refs_by_materialization_owner(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        limit=10,
        cursor=None,
    )
    assert [item.document_id for item in page.orphan_index_entries] == ["document-a"]

    repository.put_document_ref(_reference("document-a", _ownership()))
    replacement = _reference(
        "document-a",
        _ownership(binding_id="binding-b", binding_ref="knowledge-binding-b"),
    )
    repository.put_document_ref(replacement)
    with pytest.raises(
        DocumentOwnershipIndexError,
        match="reference_mismatch",
    ):
        repository.list_document_refs_by_materialization_owner(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            source_id="source-a",
            indexed_source_binding_id="binding-a",
            knowledge_source_binding_ref="knowledge-binding-a",
            limit=10,
            cursor=None,
        )
