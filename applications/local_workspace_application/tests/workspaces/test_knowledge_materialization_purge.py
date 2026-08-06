from datetime import UTC, datetime

from local_workspace_application.workspaces.knowledge_materialization_purge import (
    KnowledgeMaterializationDeletionResultV1,
    KnowledgeMaterializationPurgeRequestV1,
    KnowledgeMaterializationPurgeService,
    KnowledgeMaterializationPurgeStatusV1,
    VectorStoreKnowledgeMaterializationDeletion,
    knowledge_materialization_purge_id,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationOwnershipV1,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
)

_NOW = datetime(2026, 8, 6, tzinfo=UTC)


def _request(operation_id: str = "operation-a") -> KnowledgeMaterializationPurgeRequestV1:
    return KnowledgeMaterializationPurgeRequestV1(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        requested_lifecycle_revision=2,
        operation_id=operation_id,
    )


class _NoopDeletion:
    def delete_document_materialization(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        document_id: str,
        expected_ownership: object,
    ) -> KnowledgeMaterializationDeletionResultV1:
        return KnowledgeMaterializationDeletionResultV1(already_absent=1)


def test_purge_identity_is_scope_deterministic() -> None:
    assert knowledge_materialization_purge_id(_request()) == knowledge_materialization_purge_id(
        _request("operation-b")
    )


def test_empty_purge_invalidates_then_completes_and_replays() -> None:
    store = InMemoryDocumentStore()
    repository = ManagedWorkspaceRepository(store)
    fence = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=lambda: _NOW,
        permit_id_factory=lambda: "permit-a",
    )
    fence.enable(
        tenant_id="tenant-a",
        binding_id="knowledge-binding-a",
        lifecycle_revision=1,
        lifecycle_token="token-a",
        expected_revision=None,
    )
    service = KnowledgeMaterializationPurgeService(
        repository=repository,
        publication_authority=fence,
        deletion_port=_NoopDeletion(),
        clock=lambda: _NOW,
        page_size=1,
    )
    request = _request()

    first = service.start_or_resume(request)
    assert first.state.status is KnowledgeMaterializationPurgeStatusV1.INVALIDATED
    current_fence = fence.read_fence(
        tenant_id="tenant-a", binding_id="knowledge-binding-a"
    )
    assert current_fence is not None
    assert current_fence.detached
    assert not current_fence.enabled

    result = first
    for _ in range(3):
        result = service.start_or_resume(request)
    assert result.state.status is KnowledgeMaterializationPurgeStatusV1.COMPLETED
    replay = service.start_or_resume(_request("operation-b"))
    assert replay.state.status is KnowledgeMaterializationPurgeStatusV1.COMPLETED


class _VectorManager:
    def __init__(self) -> None:
        self.matches = [
            {
                "id": "chunk-a",
                "metadata": {
                    "tenant_id": "tenant-a",
                    "workspace_id": "workspace-a",
                    "source_id": "source-a",
                    "document_id": "document-a",
                },
            }
        ]
        self.deleted: list[str] = []

    def search_by_metadata(self, *, conditions: dict[str, object], limit: int) -> list[dict[str, object]]:
        return [
            item
            for item in self.matches
            if all(item["metadata"].get(key) == value for key, value in conditions.items())
        ][:limit]

    def delete(self, ids: list[str], *, scope: object) -> None:
        self.deleted.extend(ids)
        self.matches = [item for item in self.matches if item["id"] not in ids]


def test_vector_deletion_uses_exact_document_and_scope() -> None:
    manager = _VectorManager()
    deletion = VectorStoreKnowledgeMaterializationDeletion(manager)
    ownership = KnowledgeMaterializationOwnershipV1.connected(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        delivery_id="d" * 64,
        remote_id="remote-a",
    )

    result = deletion.delete_document_materialization(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        document_id="document-a",
        expected_ownership=ownership,
    )
    assert result.embeddings_deleted == 1
    assert manager.deleted == ["chunk-a"]
