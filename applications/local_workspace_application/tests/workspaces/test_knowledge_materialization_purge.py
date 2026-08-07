from datetime import UTC, datetime

from local_workspace_application.workspaces.connected_source_manifest import (
    ConnectedSourceMaterializationManifestEntryV1,
    ConnectedSourceMaterializationManifestRepository,
    ConnectedSourceMaterializationManifestV1,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceOperationDeliveryAccounting,
    ConnectedSourceSyncEnqueueIntent,
)
from local_workspace_application.workspaces.connected_source_recovery_ownership_index import (
    RecoveryRecordKindV1,
    recovery_ownership_index_partition,
)
from local_workspace_application.workspaces.knowledge_materialization_purge import (
    KnowledgeMaterializationDeletionResultV1,
    KnowledgeMaterializationPurgePhaseV1,
    KnowledgeMaterializationPurgeRequestV1,
    KnowledgeMaterializationPurgeService,
    KnowledgeMaterializationPurgeStatusV1,
    VectorStoreKnowledgeMaterializationDeletion,
    knowledge_materialization_purge_id,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationOwnershipV1,
    KnowledgeMaterializationVisibilityAuthorityTypeV1,
)
from local_workspace_application.workspaces.models import WorkspaceDocumentReference
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


class _RecordingDeletion:
    def __init__(self) -> None:
        self.document_ids: list[str] = []

    def delete_document_materialization(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        document_id: str,
        expected_ownership: KnowledgeMaterializationOwnershipV1,
    ) -> KnowledgeMaterializationDeletionResultV1:
        self.document_ids.append(document_id)
        return KnowledgeMaterializationDeletionResultV1(
            chunks_deleted=1,
            embeddings_deleted=1,
        )


def _reference(document_id: str, ownership: KnowledgeMaterializationOwnershipV1) -> WorkspaceDocumentReference:
    return WorkspaceDocumentReference(
        document_id=document_id,
        tenant_id=ownership.tenant_id,
        workspace_id=ownership.workspace_id,
        source_id=ownership.source_id,
        source_path=f"/{document_id}.md",
        file_name=f"{document_id}.md",
        content_hash=f"sha256:{document_id}",
        indexed_at=_NOW,
        materialization_ownership=ownership,
        visibility_authority_type=(
            KnowledgeMaterializationVisibilityAuthorityTypeV1.DELIVERY_MANIFEST
        ),
        visibility_authority_ref=ownership.delivery_id,
    )


def _run_to_terminal(
    service: KnowledgeMaterializationPurgeService,
    request: KnowledgeMaterializationPurgeRequestV1,
    *,
    max_steps: int = 400,
) -> KnowledgeMaterializationPurgeStatusV1:
    status = KnowledgeMaterializationPurgeStatusV1.PREPARING
    for _ in range(max_steps):
        status = service.start_or_resume(request).state.status
        if status in {
            KnowledgeMaterializationPurgeStatusV1.COMPLETED,
            KnowledgeMaterializationPurgeStatusV1.FAILED,
        }:
            return status
    return status


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
    for _ in range(20):
        result = service.start_or_resume(request)
        if result.state.status is KnowledgeMaterializationPurgeStatusV1.COMPLETED:
            break
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


def test_purge_pages_exact_document_ownership_and_preserves_other_binding() -> None:
    store = InMemoryDocumentStore(cursor_secret=b"purge-cursor-secret")
    repository = ManagedWorkspaceRepository(store)
    deletion = _RecordingDeletion()
    target_ownership = KnowledgeMaterializationOwnershipV1.connected(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        delivery_id="a" * 64,
        remote_id="remote-a",
    )
    other_ownership = target_ownership.model_copy(
        update={
            "indexed_source_binding_id": "binding-b",
            "knowledge_source_binding_ref": "knowledge-binding-b",
        }
    )
    for index in range(3):
        repository.put_document_ref(
            _reference(f"document-{index}", target_ownership.model_copy(
                update={"remote_id": f"remote-{index}"}
            ))
        )
    repository.put_document_ref(_reference("other-document", other_ownership))
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
        deletion_port=deletion,
        clock=lambda: _NOW,
        page_size=1,
    )

    assert _run_to_terminal(service, _request()) is (
        KnowledgeMaterializationPurgeStatusV1.COMPLETED
    )
    assert set(deletion.document_ids) == {
        "document-0",
        "document-1",
        "document-2",
    }
    assert repository.get_document_ref(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        document_id="other-document",
    ) is not None


def test_purge_deletes_complete_recovery_records_without_operation_scan() -> None:
    store = InMemoryDocumentStore(cursor_secret=b"purge-recovery-secret")
    repository = ManagedWorkspaceRepository(store)
    delivery_id = "b" * 64
    repository.put_connected_source_sync_enqueue_intent(
        ConnectedSourceSyncEnqueueIntent(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            source_id="source-a",
            indexed_source_binding_id="binding-a",
            knowledge_source_binding_ref="knowledge-binding-a",
            operation_id="operation-recovery",
            enqueue_generation=1,
            updated_at=_NOW,
            ownership_classification="COMPLETE_OWNERSHIP",
        )
    )
    repository.put_connected_source_delivery_accounting_if_absent(
        ConnectedSourceOperationDeliveryAccounting(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            source_id="source-a",
            indexed_source_binding_id="binding-a",
            knowledge_source_binding_ref="knowledge-binding-a",
            operation_id="operation-recovery",
            delivery_id=delivery_id,
            documents_indexed=1,
            documents_unchanged=0,
            items_failed=0,
            accounted_at=_NOW,
            ownership_classification="COMPLETE_OWNERSHIP",
        )
    )
    fence = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=lambda: _NOW,
        permit_id_factory=lambda: "permit-recovery",
    )
    fence.enable(
        tenant_id="tenant-a",
        binding_id="knowledge-binding-a",
        lifecycle_revision=1,
        lifecycle_token="token-recovery",
        expected_revision=None,
    )
    service = KnowledgeMaterializationPurgeService(
        repository=repository,
        publication_authority=fence,
        deletion_port=_NoopDeletion(),
        clock=lambda: _NOW,
        page_size=1,
    )

    assert _run_to_terminal(service, _request()) is (
        KnowledgeMaterializationPurgeStatusV1.COMPLETED
    )
    assert repository.get_connected_source_sync_enqueue_intent(
        tenant_id="tenant-a",
        operation_id="operation-recovery",
    ) is None
    assert repository.get_connected_source_delivery_accounting(
        tenant_id="tenant-a",
        operation_id="operation-recovery",
        delivery_id=delivery_id,
    ) is None


def _enable_fence(store: InMemoryDocumentStore, *, binding_id: str = 'knowledge-binding-a') -> DocumentStoreKnowledgeSyncPublicationFenceRepository:
    fence = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=lambda: _NOW,
        permit_id_factory=lambda: 'permit-a',
    )
    fence.enable(
        tenant_id='tenant-a',
        binding_id=binding_id,
        lifecycle_revision=1,
        lifecycle_token='token-a',
        expected_revision=None,
    )
    return fence


def _manifest_with_entries(count: int, *, delivery_id: str = 'c' * 64) -> ConnectedSourceMaterializationManifestV1:
    entries = tuple(
        ConnectedSourceMaterializationManifestEntryV1(
            remote_id=f'remote-{index:04d}',
            document_id=f'document-{index:04d}',
            materialization_generation='generation-1',
            content_hash=f'content-{index:04d}',
        )
        for index in range(count)
    )
    return ConnectedSourceMaterializationManifestV1(
        tenant_id='tenant-a',
        workspace_id='workspace-a',
        source_id='source-a',
        indexed_source_binding_id='binding-a',
        knowledge_source_binding_ref='knowledge-binding-a',
        delivery_id=delivery_id,
        materialization_sequence=1,
        binding_configuration_version=1,
        publication_fence_revision=1,
        publication_fence_token_fingerprint='d' * 64,
        document_entries=entries,
        payload_fingerprint='e' * 64,
        committed_at=_NOW,
    )


def test_manifest_entry_paging_1000_by_137_and_crash_points() -> None:
    store = InMemoryDocumentStore(cursor_secret=b'purge-manifest-secret')
    repository = ManagedWorkspaceRepository(store)
    fence = _enable_fence(store)
    manifest_repository = ConnectedSourceMaterializationManifestRepository(store)
    manifest = _manifest_with_entries(1000)
    manifest_repository._put_immutable(manifest)
    recording = _RecordingDeletion()
    service = KnowledgeMaterializationPurgeService(
        repository=repository,
        publication_authority=fence,
        deletion_port=recording,
        clock=lambda: _NOW,
        page_size=137,
    )
    request = _request()
    entry_batches: list[int] = []
    crash_after = {1, 136, 137, 500, 999}
    crashed: set[int] = set()
    for _ in range(80):
        before = len(recording.document_ids)
        state_before = service.start_or_resume(request).state
        processed = len(recording.document_ids) - before
        if processed:
            entry_batches.append(processed)
            assert processed <= 137
            total = len(recording.document_ids)
            for point in sorted(crash_after):
                if point <= total:
                    crashed.add(point)
        if state_before.status in {
            KnowledgeMaterializationPurgeStatusV1.COMPLETED,
            KnowledgeMaterializationPurgeStatusV1.FAILED,
        }:
            break
    assert len(recording.document_ids) == 1000
    assert len(set(recording.document_ids)) == 1000
    assert len(entry_batches) >= 8
    assert max(entry_batches) <= 137
    assert crash_after <= crashed
    assert state_before.status is KnowledgeMaterializationPurgeStatusV1.COMPLETED
    assert store.query(
        manifest_repository._immutable_partition('tenant-a'),
        limit=1,
        row_key_prefix=manifest_repository._scope_prefix(
            workspace_id='workspace-a',
            source_id='source-a',
            indexed_source_binding_id='binding-a',
        )
        + 'manifest:',
    ).documents == ()


def test_full_bound_page_size_seven_with_100_manifest_entries() -> None:
    store = InMemoryDocumentStore(cursor_secret=b'purge-bound-secret')
    repository = ManagedWorkspaceRepository(store)
    fence = _enable_fence(store)
    ownership = KnowledgeMaterializationOwnershipV1.connected(
        tenant_id='tenant-a',
        workspace_id='workspace-a',
        source_id='source-a',
        indexed_source_binding_id='binding-a',
        knowledge_source_binding_ref='knowledge-binding-a',
        delivery_id='f' * 64,
        remote_id='remote-doc',
        materialization_generation='generation-1',
        materialization_sequence=1,
    )
    for index in range(3):
        repository.put_document_ref(
            _reference(f'owned-{index}', ownership.model_copy(update={'remote_id': f'remote-{index}'}))
        )
    manifest_repository = ConnectedSourceMaterializationManifestRepository(store)
    manifest_repository._put_immutable(_manifest_with_entries(100, delivery_id='f' * 64))
    recording = _RecordingDeletion()
    service = KnowledgeMaterializationPurgeService(
        repository=repository,
        publication_authority=fence,
        deletion_port=recording,
        clock=lambda: _NOW,
        page_size=7,
    )
    request = _request()
    for _ in range(80):
        before = len(recording.document_ids)
        result = service.start_or_resume(request)
        processed = len(recording.document_ids) - before
        assert processed <= 7
        if result.state.status in {
            KnowledgeMaterializationPurgeStatusV1.COMPLETED,
            KnowledgeMaterializationPurgeStatusV1.FAILED,
        }:
            break
    assert result.state.status is KnowledgeMaterializationPurgeStatusV1.COMPLETED


def test_recovery_scale_enumerates_only_target_ownership_prefix() -> None:
    store = InMemoryDocumentStore(cursor_secret=b'purge-scale-secret')
    repository = ManagedWorkspaceRepository(store)
    fence = _enable_fence(store)
    queries: list[tuple[str, str | None]] = []
    original_query = store.query

    def spy_query(partition_key: str, *args: object, **kwargs: object):
        queries.append((partition_key, kwargs.get('row_key_prefix')))
        return original_query(partition_key, *args, **kwargs)

    store.query = spy_query  # type: ignore[method-assign]
    for index in range(10):
        repository.put_connected_source_sync_enqueue_intent(
            ConnectedSourceSyncEnqueueIntent(
                tenant_id='tenant-a',
                workspace_id='workspace-a',
                source_id='source-a',
                indexed_source_binding_id='binding-a',
                knowledge_source_binding_ref='knowledge-binding-a',
                operation_id=f'target-{index}',
                enqueue_generation=1,
                updated_at=_NOW,
                ownership_classification='COMPLETE_OWNERSHIP',
            )
        )
    for index in range(5000):
        repository.put_connected_source_sync_enqueue_intent(
            ConnectedSourceSyncEnqueueIntent(
                tenant_id='tenant-a',
                workspace_id='workspace-other',
                source_id='source-other',
                indexed_source_binding_id='binding-other',
                knowledge_source_binding_ref='knowledge-other',
                operation_id=f'unrelated-{index}',
                enqueue_generation=1,
                updated_at=_NOW,
                ownership_classification='COMPLETE_OWNERSHIP',
            )
        )
    service = KnowledgeMaterializationPurgeService(
        repository=repository,
        publication_authority=fence,
        deletion_port=_NoopDeletion(),
        clock=lambda: _NOW,
        page_size=7,
    )
    assert _run_to_terminal(service, _request()) is KnowledgeMaterializationPurgeStatusV1.COMPLETED
    recovery_partition = recovery_ownership_index_partition('tenant-a')
    recovery_queries = [
        prefix for partition, prefix in queries if partition == recovery_partition
    ]
    assert recovery_queries
    assert all(isinstance(prefix, str) and prefix.startswith('owner:') for prefix in recovery_queries)
    enqueue_partition = 'lkw.managed_workspace:tenant-a:connected_source_sync_enqueue'
    assert not any(partition == enqueue_partition for partition, _ in queries)
    assert repository.get_connected_source_sync_enqueue_intent(
        tenant_id='tenant-a',
        operation_id='unrelated-0',
    ) is not None
    assert repository.list_connected_source_recovery_records_by_owner(
        tenant_id='tenant-a',
        workspace_id='workspace-a',
        source_id='source-a',
        indexed_source_binding_id='binding-a',
        knowledge_source_binding_ref='knowledge-binding-a',
        record_kind=RecoveryRecordKindV1.ENQUEUE_INTENT,
        limit=1,
    ).index_entries == ()


def test_concurrent_manifest_workers_do_not_regress_cursor() -> None:
    store = InMemoryDocumentStore(cursor_secret=b'purge-concurrent-secret')
    repository = ManagedWorkspaceRepository(store)
    fence = _enable_fence(store)
    manifest_repository = ConnectedSourceMaterializationManifestRepository(store)
    manifest_repository._put_immutable(_manifest_with_entries(50, delivery_id='b' * 64))
    service = KnowledgeMaterializationPurgeService(
        repository=repository,
        publication_authority=fence,
        deletion_port=_NoopDeletion(),
        clock=lambda: _NOW,
        page_size=10,
    )
    request = _request()
    offsets: list[int] = []
    for _ in range(40):
        first = service.start_or_resume(request).state
        second = service.start_or_resume(request).state
        for state in (first, second):
            if (
                state.cursor is not None
                and state.cursor.phase is KnowledgeMaterializationPurgePhaseV1.MANIFESTS
                and state.cursor.manifest_entry_offset is not None
            ):
                offsets.append(state.cursor.manifest_entry_offset)
        if first.status is KnowledgeMaterializationPurgeStatusV1.COMPLETED:
            break
        if second.status is KnowledgeMaterializationPurgeStatusV1.COMPLETED:
            break
    assert offsets == sorted(offsets)
    assert service.start_or_resume(request).state.status is (
        KnowledgeMaterializationPurgeStatusV1.COMPLETED
    )


def test_manifest_identity_mismatch_blocks() -> None:
    store = InMemoryDocumentStore(cursor_secret=b'purge-mismatch-manifest')
    repository = ManagedWorkspaceRepository(store)
    fence = _enable_fence(store)
    manifest_repository = ConnectedSourceMaterializationManifestRepository(store)
    manifest = _manifest_with_entries(3, delivery_id='a' * 64)
    manifest_repository._put_immutable(manifest)
    service = KnowledgeMaterializationPurgeService(
        repository=repository,
        publication_authority=fence,
        deletion_port=_NoopDeletion(),
        clock=lambda: _NOW,
        page_size=1,
    )
    request = _request()
    while True:
        state = service.start_or_resume(request).state
        if (
            state.cursor is not None
            and state.cursor.phase is KnowledgeMaterializationPurgePhaseV1.MANIFESTS
            and state.cursor.current_manifest_id is not None
        ):
            break
        if state.status in {
            KnowledgeMaterializationPurgeStatusV1.COMPLETED,
            KnowledgeMaterializationPurgeStatusV1.FAILED,
        }:
            raise AssertionError('manifest phase not reached')
    store.delete(
        manifest_repository._immutable_partition('tenant-a'),
        state.cursor.current_manifest_row_key or '',
    )
    failed = service.start_or_resume(request).state
    assert failed.status is KnowledgeMaterializationPurgeStatusV1.FAILED
    assert failed.last_error_code == 'BLOCKED_CORRUPT_STATE'
