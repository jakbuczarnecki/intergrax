"""Final publication-fence closeout invariant proofs (targeted gaps only)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from local_workspace_application.workspaces.connected_source_manifest import (
    ConnectedSourceMaterializationManifestConflict,
    ConnectedSourceMaterializationManifestEntryV1,
    ConnectedSourceMaterializationManifestRepository,
    ConnectedSourceMaterializationManifestV1,
    ManifestCommitStatus,
)
from local_workspace_application.workspaces.connected_source_purge_completion_contracts import (
    migration_gate_for_new_ownership_complete_binding,
    migration_gate_required,
)
from local_workspace_application.workspaces.knowledge_materialization_purge import (
    KnowledgeMaterializationDeletionResultV1,
    KnowledgeMaterializationPurgeRequestV1,
    KnowledgeMaterializationPurgeService,
    KnowledgeMaterializationPurgeStatusV1,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationActivePointerV1,
    KnowledgeMaterializationOwnershipV1,
    KnowledgeMaterializationVisibilityAuthorityTypeV1,
    RepositoryKnowledgeMaterializationVisibility,
)
from local_workspace_application.workspaces.models import (
    WorkspaceDocumentReference,
    WorkspaceSource,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
    KnowledgeSyncCommittedPublicationV1,
    KnowledgeSyncPublicationFenceConflict,
    KnowledgeSyncPublicationFenceV1,
    KnowledgeSyncPublicationInProgress,
    publication_commit_id,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_SOURCE = "source-a"
_INDEXED = "binding-a"
_BINDING = "knowledge-binding-a"
_TOKEN = "token-a"


class _NoopDeletion:
    def delete_document_materialization(self, **kwargs: object) -> KnowledgeMaterializationDeletionResultV1:
        return KnowledgeMaterializationDeletionResultV1(
            chunks_deleted=0,
            embeddings_deleted=0,
        )


def _entry(remote_id: str, document_id: str) -> ConnectedSourceMaterializationManifestEntryV1:
    return ConnectedSourceMaterializationManifestEntryV1(
        remote_id=remote_id,
        document_id=document_id,
        materialization_generation="generation-1",
        content_hash=f"content-{remote_id}",
    )


def _manifest(
    *,
    sequence: int,
    delivery_id: str,
    remote_id: str = "remote-a",
    document_id: str = "document-a",
    entries: tuple[ConnectedSourceMaterializationManifestEntryV1, ...] | None = None,
) -> ConnectedSourceMaterializationManifestV1:
    return ConnectedSourceMaterializationManifestV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
        delivery_id=delivery_id,
        materialization_sequence=sequence,
        binding_configuration_version=1,
        publication_fence_revision=1,
        publication_fence_token_fingerprint="a" * 64,
        document_entries=(
            (_entry(remote_id, document_id),) if entries is None else entries
        ),
        payload_fingerprint=delivery_id,
        committed_at=_NOW,
    )


def _enable(
    store: InMemoryDocumentStore,
    *,
    permit_id: str = "permit-a",
    clock=None,
) -> tuple[
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
    KnowledgeSyncPublicationFenceV1,
]:
    authority = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=clock or (lambda: _NOW),
        permit_id_factory=lambda: permit_id,
    )
    fence = KnowledgeSyncPublicationFenceV1(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        lifecycle_revision=1,
        lifecycle_token=_TOKEN,
        enabled=True,
        detached=False,
    )
    authority.write_fence(fence, expected_revision=None)
    return authority, fence


def _acquire(authority, fence, *, owner_id: str = "owner-a", ttl_seconds: int = 60):
    permit = authority.acquire_publication_permit(
        tenant_id=fence.tenant_id,
        binding_id=fence.binding_id,
        expected_revision=fence.lifecycle_revision,
        expected_token=fence.lifecycle_token,
        owner_id=owner_id,
        ttl_seconds=ttl_seconds,
    )
    assert permit is not None
    return permit


def _purge_request(operation_id: str = "operation-a") -> KnowledgeMaterializationPurgeRequestV1:
    return KnowledgeMaterializationPurgeRequestV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
        requested_lifecycle_revision=2,
        operation_id=operation_id,
    )


def _seed_migration_cleared(repository: ManagedWorkspaceRepository) -> None:
    repository.put_connected_source_recovery_migration_gate(
        migration_gate_for_new_ownership_complete_binding(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_INDEXED,
            knowledge_source_binding_ref=_BINDING,
            cleared_at=_NOW,
        )
    )


def _run_purge(
    service: KnowledgeMaterializationPurgeService,
    request: KnowledgeMaterializationPurgeRequestV1,
    *,
    max_steps: int = 200,
):
    state = None
    for _ in range(max_steps):
        state = service.start_or_resume(request).state
        if state.status in {
            KnowledgeMaterializationPurgeStatusV1.COMPLETED,
            KnowledgeMaterializationPurgeStatusV1.FAILED,
        }:
            return state
    assert state is not None
    return state


def test_prepared_manifest_and_orphan_commit_node_are_invisible_before_authority_cas() -> None:
    class _CrashBeforeAuthorityCAS(InMemoryDocumentStore):
        crash = False

        def replace_if_match(self, *, expected, replacement):
            if (
                self.crash
                and expected.row_key == f"binding:{_BINDING}"
                and replacement.data.get("committed_publication") is not None
                and expected.data.get("committed_publication") is None
            ):
                self.crash = False
                raise RuntimeError("injected crash before authority CAS")
            return super().replace_if_match(expected=expected, replacement=replacement)

    store = _CrashBeforeAuthorityCAS()
    authority, fence = _enable(store)
    permit = _acquire(authority, fence)
    repository = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=authority,
    )
    manifest = _manifest(sequence=1, delivery_id="a" * 64)
    store.crash = True
    with pytest.raises(RuntimeError, match="before authority CAS"):
        repository.commit(
            manifest,
            expected_fence=fence,
            publication_permit=permit,
        )

    assert authority.read_publication_head(tenant_id=_TENANT, binding_id=_BINDING) is None
    assert (
        repository.get_committed_for_delivery(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_INDEXED,
            delivery_id=manifest.delivery_id,
        )
        is None
    )
    assert repository.get_prepared_for_delivery(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        delivery_id=manifest.delivery_id,
    ) == manifest
    descriptor = KnowledgeSyncCommittedPublicationV1(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        delivery_id=manifest.delivery_id,
        materialization_sequence=1,
        manifest_id=manifest.manifest_id,
        manifest_fingerprint=manifest.manifest_fingerprint,
        committed_at=_NOW,
    )
    orphan_id = publication_commit_id(descriptor, previous_commit_id=None)
    node = authority.read_publication_commit_node(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        commit_id=orphan_id,
    )
    assert node.commit_id == orphan_id
    assert authority.read_committed_publication(tenant_id=_TENANT, binding_id=_BINDING) is None


def test_same_delivery_replay_is_idempotent_without_second_history_node() -> None:
    store = InMemoryDocumentStore()
    authority, fence = _enable(store)
    permit = _acquire(authority, fence)
    repository = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=authority,
    )
    manifest = _manifest(sequence=1, delivery_id="a" * 64)
    assert (
        repository.commit(manifest, expected_fence=fence, publication_permit=permit)
        is ManifestCommitStatus.COMMITTED
    )
    head = authority.read_publication_head(tenant_id=_TENANT, binding_id=_BINDING)
    assert head is not None
    assert (
        repository.commit(manifest, expected_fence=fence, publication_permit=permit)
        is ManifestCommitStatus.REPLAYED
    )
    chain = authority.list_committed_publications(tenant_id=_TENANT, binding_id=_BINDING)
    assert len(chain) == 1
    assert chain[0].publication_commit_id == head.publication_commit_id


def test_two_independent_publishers_one_cas_winner_no_visible_fork() -> None:
    store = InMemoryDocumentStore()
    authority_a, fence = _enable(store, permit_id="permit-shared")
    authority_b = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=lambda: _NOW,
        permit_id_factory=lambda: "unused",
    )
    permit = _acquire(authority_a, fence, owner_id="shared-owner")
    repo_a = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=authority_a,
    )
    repo_b = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=authority_b,
    )
    first = _manifest(sequence=1, delivery_id="a" * 64, document_id="document-1")
    assert (
        repo_a.commit(first, expected_fence=fence, publication_permit=permit)
        is ManifestCommitStatus.COMMITTED
    )
    winner = _manifest(
        sequence=2,
        delivery_id="b" * 64,
        remote_id="remote-b",
        document_id="document-b",
    )
    loser = _manifest(
        sequence=2,
        delivery_id="c" * 64,
        remote_id="remote-c",
        document_id="document-c",
    )
    assert (
        repo_a.commit(winner, expected_fence=fence, publication_permit=permit)
        is ManifestCommitStatus.COMMITTED
    )
    with pytest.raises(ConnectedSourceMaterializationManifestConflict):
        repo_b.commit(loser, expected_fence=fence, publication_permit=permit)

    chain = authority_b.list_committed_publications(tenant_id=_TENANT, binding_id=_BINDING)
    assert [item.delivery_id for item in chain] == ["b" * 64, "a" * 64]
    assert repo_b.get_committed_for_delivery(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        delivery_id="c" * 64,
    ) is None
    head = authority_a.read_publication_head(tenant_id=_TENANT, binding_id=_BINDING)
    assert head is not None
    assert head.delivery_id == "b" * 64


def test_disable_beats_stale_publisher_and_publisher_beats_disable_ordering() -> None:
    store = InMemoryDocumentStore()
    now = {"value": _NOW}
    publisher = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=lambda: now["value"],
        permit_id_factory=lambda: "permit-pub",
    )
    lifecycle = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=lambda: now["value"],
        permit_id_factory=lambda: "unused",
    )
    fence = KnowledgeSyncPublicationFenceV1(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        lifecycle_revision=1,
        lifecycle_token=_TOKEN,
        enabled=True,
        detached=False,
    )
    publisher.write_fence(fence, expected_revision=None)
    permit = _acquire(publisher, fence)
    repository = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=publisher,
    )
    page = _manifest(sequence=1, delivery_id="a" * 64)
    assert (
        repository.commit(page, expected_fence=fence, publication_permit=permit)
        is ManifestCommitStatus.COMMITTED
    )
    # Active permit blocks disable.
    with pytest.raises(KnowledgeSyncPublicationInProgress):
        lifecycle.disable(
            tenant_id=_TENANT,
            binding_id=_BINDING,
            lifecycle_revision=2,
            lifecycle_token="token-b",
            expected_revision=1,
        )
    assert publisher.release_publication_permit(permit) is True
    disabled = lifecycle.disable(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        lifecycle_revision=2,
        lifecycle_token="token-b",
        expected_revision=1,
    )
    assert disabled.enabled is False
    with pytest.raises(KnowledgeSyncPublicationFenceConflict):
        publisher.commit_publication_under_permit(
            expected_fence=fence,
            publication_permit=permit,
            publication_descriptor=KnowledgeSyncCommittedPublicationV1(
                tenant_id=_TENANT,
                binding_id=_BINDING,
                workspace_id=_WORKSPACE,
                source_id=_SOURCE,
                indexed_source_binding_id=_INDEXED,
                delivery_id="b" * 64,
                materialization_sequence=2,
                manifest_id="m" * 64,
                manifest_fingerprint="f" * 64,
                committed_at=_NOW,
            ),
        )
    # Historical committed head remains durable, but fence disable blocks new authority.
    head = publisher.read_publication_head(tenant_id=_TENANT, binding_id=_BINDING)
    assert head is not None
    assert head.delivery_id == "a" * 64


def test_purge_vs_active_permit_and_history_cleanup_restart_and_resurrection() -> None:
    store = InMemoryDocumentStore(cursor_secret=b"closeout-purge")
    repository = ManagedWorkspaceRepository(store)
    _seed_migration_cleared(repository)
    authority, fence = _enable(store)
    permit = _acquire(authority, fence)
    manifests = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=authority,
    )
    page_one = _manifest(
        sequence=1,
        delivery_id="a" * 64,
        entries=(_entry("remote-a", "document-a"), _entry("remote-b", "document-b")),
    )
    page_two = _manifest(
        sequence=2,
        delivery_id="b" * 64,
        remote_id="remote-c",
        document_id="document-c",
    )
    page_three = _manifest(
        sequence=3,
        delivery_id="c" * 64,
        remote_id="remote-a",
        document_id="document-a-v2",
    )
    for page in (page_one, page_two, page_three):
        assert (
            manifests.commit(page, expected_fence=fence, publication_permit=permit)
            is ManifestCommitStatus.COMMITTED
        )
    purge_blocked = KnowledgeMaterializationPurgeService(
        repository=repository,
        publication_authority=DocumentStoreKnowledgeSyncPublicationFenceRepository(
            store,
            clock=lambda: _NOW,
            permit_id_factory=lambda: "purge-blocked",
        ),
        deletion_port=_NoopDeletion(),
        clock=lambda: _NOW,
        page_size=2,
    )
    blocked = purge_blocked.start_or_resume(_purge_request("blocked")).state
    assert blocked.status is KnowledgeMaterializationPurgeStatusV1.FAILED
    assert blocked.last_error_code == "publication_in_progress"

    assert authority.release_publication_permit(permit) is True
    old_token = fence.lifecycle_token
    old_revision = fence.lifecycle_revision
    prepared_orphan = page_three

    # Restart proof: reconstruct services across steps.
    states = []
    request = _purge_request("purge-main")
    for _ in range(120):
        service = KnowledgeMaterializationPurgeService(
            repository=ManagedWorkspaceRepository(store),
            publication_authority=DocumentStoreKnowledgeSyncPublicationFenceRepository(
                store,
                clock=lambda: _NOW,
                permit_id_factory=lambda: "purge-restart",
            ),
            deletion_port=_NoopDeletion(),
            clock=lambda: _NOW,
            page_size=1,
        )
        state = service.start_or_resume(request).state
        states.append(state)
        if state.status in {
            KnowledgeMaterializationPurgeStatusV1.COMPLETED,
            KnowledgeMaterializationPurgeStatusV1.FAILED,
        }:
            break
    final = states[-1]
    assert final.status is KnowledgeMaterializationPurgeStatusV1.COMPLETED
    assert final.counters.publication_nodes_deleted == 3
    assert final.counters.manifests_deleted == 3

    restarted = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=lambda: _NOW,
    )
    tombstone = restarted.read_fence(tenant_id=_TENANT, binding_id=_BINDING)
    assert tombstone is not None
    assert tombstone.detached is True
    assert tombstone.enabled is False
    assert restarted.read_publication_head(tenant_id=_TENANT, binding_id=_BINDING) is None
    assert restarted.acquire_publication_permit(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        expected_revision=old_revision,
        expected_token=old_token,
        owner_id="stale-worker",
        ttl_seconds=30,
    ) is None
    assert restarted.acquire_publication_permit(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        expected_revision=tombstone.lifecycle_revision,
        expected_token=tombstone.lifecycle_token,
        owner_id="fresh-worker",
        ttl_seconds=30,
    ) is None
    with pytest.raises(KnowledgeSyncPublicationFenceConflict):
        restarted.commit_publication_under_permit(
            expected_fence=fence,
            publication_permit=permit,
            publication_descriptor=KnowledgeSyncCommittedPublicationV1(
                tenant_id=_TENANT,
                binding_id=_BINDING,
                workspace_id=_WORKSPACE,
                source_id=_SOURCE,
                indexed_source_binding_id=_INDEXED,
                delivery_id=prepared_orphan.delivery_id,
                materialization_sequence=prepared_orphan.materialization_sequence,
                manifest_id=prepared_orphan.manifest_id,
                manifest_fingerprint=prepared_orphan.manifest_fingerprint,
                committed_at=_NOW,
            ),
        )


def test_query_visibility_after_disable_and_purge_and_legacy_block() -> None:
    store = InMemoryDocumentStore(cursor_secret=b"closeout-visibility")
    repository = ManagedWorkspaceRepository(store)
    repository.put_source(
        WorkspaceSource(
            source_id=_SOURCE,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_type=WorkspaceSourceType.CONNECTED_SOURCE,
            knowledge_configuration_creation_mutation_id="mutation-a",
            knowledge_configuration_visibility_revision=1,
            created_at=_NOW,
        )
    )
    authority, fence = _enable(store)
    permit = _acquire(authority, fence)
    manifests = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=authority,
    )
    page = _manifest(sequence=1, delivery_id="a" * 64)
    assert (
        manifests.commit(page, expected_fence=fence, publication_permit=permit)
        is ManifestCommitStatus.COMMITTED
    )
    ownership = KnowledgeMaterializationOwnershipV1.connected(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
        delivery_id=page.delivery_id,
        remote_id="remote-a",
        materialization_generation="generation-1",
        materialization_sequence=1,
    )
    repository.put_document_ref(
        WorkspaceDocumentReference(
            document_id="document-a",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            source_path="/document-a.md",
            file_name="document-a.md",
            content_hash="sha256:document-a",
            indexed_at=_NOW,
            materialization_ownership=ownership,
            visibility_authority_type=(
                KnowledgeMaterializationVisibilityAuthorityTypeV1.DELIVERY_MANIFEST
            ),
            visibility_authority_ref=page.delivery_id,
        )
    )
    # Stale active pointer alone must not override fence/manifest authority.
    repository.put_active_materialization_pointer_if_absent(
        KnowledgeMaterializationActivePointerV1.for_ownership(
            ownership=ownership,
            document_id="document-a",
            materialization_revision=1,
            committed_at=_NOW,
        )
    )
    visibility = RepositoryKnowledgeMaterializationVisibility(repository)
    assert visibility.is_visible(ownership=ownership, document_id="document-a") is True

    assert authority.release_publication_permit(permit) is True
    authority.disable(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        lifecycle_revision=2,
        lifecycle_token="token-disabled",
        expected_revision=1,
    )
    assert visibility.is_visible(ownership=ownership, document_id="document-a") is False

    repository.put_connected_source_recovery_migration_gate(
        migration_gate_required(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_INDEXED,
            knowledge_source_binding_ref=_BINDING,
        )
    )
    purge = KnowledgeMaterializationPurgeService(
        repository=repository,
        publication_authority=DocumentStoreKnowledgeSyncPublicationFenceRepository(
            store,
            clock=lambda: _NOW,
        ),
        deletion_port=_NoopDeletion(),
        clock=lambda: _NOW,
        page_size=5,
    )
    failed = _run_purge(purge, _purge_request("legacy"))
    assert failed.status is KnowledgeMaterializationPurgeStatusV1.FAILED
    assert failed.last_error_code == "BLOCKED_LEGACY_MIGRATION"
    tombstone = authority.read_fence(tenant_id=_TENANT, binding_id=_BINDING)
    assert tombstone is not None
    assert tombstone.detached is True
    assert visibility.is_visible(ownership=ownership, document_id="document-a") is False
    assert authority.acquire_publication_permit(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        expected_revision=tombstone.lifecycle_revision,
        expected_token=tombstone.lifecycle_token,
        owner_id="legacy-worker",
        ttl_seconds=30,
    ) is None


def test_permit_expiry_cannot_convert_prepared_records_into_authority() -> None:
    store = InMemoryDocumentStore()
    now = {"value": _NOW}
    authority, fence = _enable(store, clock=lambda: now["value"])
    permit = _acquire(authority, fence, ttl_seconds=1)
    repository = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=authority,
    )
    manifest = _manifest(sequence=1, delivery_id="a" * 64)
    repository._put_immutable(manifest)
    repository._put_delivery_index(manifest)
    repository._put_remote_candidates(manifest)
    now["value"] = _NOW + timedelta(seconds=2)
    with pytest.raises(KnowledgeSyncPublicationFenceConflict, match="expired"):
        authority.commit_publication_under_permit(
            expected_fence=fence,
            publication_permit=permit,
            publication_descriptor=KnowledgeSyncCommittedPublicationV1(
                tenant_id=_TENANT,
                binding_id=_BINDING,
                workspace_id=_WORKSPACE,
                source_id=_SOURCE,
                indexed_source_binding_id=_INDEXED,
                delivery_id=manifest.delivery_id,
                materialization_sequence=1,
                manifest_id=manifest.manifest_id,
                manifest_fingerprint=manifest.manifest_fingerprint,
                committed_at=_NOW,
            ),
        )
    assert authority.read_publication_head(tenant_id=_TENANT, binding_id=_BINDING) is None
    assert (
        repository.get_committed_for_delivery(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_INDEXED,
            delivery_id=manifest.delivery_id,
        )
        is None
    )
