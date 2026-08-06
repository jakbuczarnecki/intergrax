from datetime import UTC, datetime

import pytest
from local_workspace_application.workspaces.connected_source_manifest import (
    ConnectedSourceMaterializationManifestConflict,
    ConnectedSourceMaterializationManifestEntryV1,
    ConnectedSourceMaterializationManifestRepository,
    ConnectedSourceMaterializationManifestV1,
    ManifestCommitStatus,
)
from pydantic import ValidationError

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
    KnowledgeSyncPublicationFenceV1,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 6, 8, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_SOURCE = "source-a"
_INDEXED = "indexed-a"
_BINDING = "binding-a"
_TOKEN = "lifecycle-token"


def _entry(remote_id: str, document_id: str) -> ConnectedSourceMaterializationManifestEntryV1:
    return ConnectedSourceMaterializationManifestEntryV1(
        remote_id=remote_id,
        document_id=document_id,
        materialization_generation="generation-1",
        content_hash="content-" + remote_id,
    )


def _manifest(
    *,
    sequence: int,
    delivery_id: str,
    document_id: str,
    remote_id: str = "remote-a",
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
        document_entries=entries or (_entry(remote_id, document_id),),
        payload_fingerprint=delivery_id,
        committed_at=_NOW,
    )


def _authority(store: InMemoryDocumentStore):
    authority = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        permit_id_factory=lambda: "permit-a",
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
    permit = authority.acquire_publication_permit(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        expected_revision=1,
        expected_token=_TOKEN,
        owner_id="owner-a",
        ttl_seconds=60,
    )
    assert permit is not None
    return authority, fence, permit


def test_manifest_is_strict_bounded_and_deterministic() -> None:
    with pytest.raises(ValidationError, match="entries_not_deterministic"):
        _manifest(
            sequence=1,
            delivery_id="a" * 64,
            document_id="document-a",
            entries=(
                _entry("remote-b", "document-b"),
                _entry("remote-a", "document-a"),
            ),
        )


def test_manifest_cas_orders_sequences_and_rejects_same_sequence_conflict() -> None:
    store = InMemoryDocumentStore()
    authority, fence, permit = _authority(store)
    repository = ConnectedSourceMaterializationManifestRepository(store)

    validate = lambda expected, current: (
        None
        if authority.is_current_publication_permit(permit=current)
        else (_ for _ in ()).throw(RuntimeError("permit_lost"))
    )
    first = _manifest(sequence=1, delivery_id="a" * 64, document_id="document-a")
    assert (
        repository.commit(
            first,
            expected_fence=fence,
            publication_permit=permit,
            validate_publication=validate,
        )
        is ManifestCommitStatus.COMMITTED
    )
    second = _manifest(sequence=2, delivery_id="b" * 64, document_id="document-b")
    assert (
        repository.commit(
            second,
            expected_fence=fence,
            publication_permit=permit,
            validate_publication=validate,
        )
        is ManifestCommitStatus.COMMITTED
    )
    assert (
        repository.commit(
            first,
            expected_fence=fence,
            publication_permit=permit,
            validate_publication=validate,
        )
        is ManifestCommitStatus.STALE
    )
    conflicting = _manifest(
        sequence=2,
        delivery_id="c" * 64,
        document_id="document-c",
    )
    with pytest.raises(ConnectedSourceMaterializationManifestConflict):
        repository.commit(
            conflicting,
            expected_fence=fence,
            publication_permit=permit,
            validate_publication=validate,
        )


def test_manifest_commit_requires_current_permit() -> None:
    store = InMemoryDocumentStore()
    authority, fence, permit = _authority(store)
    assert authority.release_publication_permit(permit) is True
    repository = ConnectedSourceMaterializationManifestRepository(store)
    with pytest.raises(RuntimeError, match="permit_lost"):
        repository.commit(
            _manifest(sequence=1, delivery_id="a" * 64, document_id="document-a"),
            expected_fence=fence,
            publication_permit=permit,
            validate_publication=lambda expected, current: (
                None
                if authority.is_current_publication_permit(permit=current)
                else (_ for _ in ()).throw(RuntimeError("permit_lost"))
            ),
        )
    assert repository.get_current(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
    ) is None


def test_manifest_visibility_is_linearized_by_fence_record_cas() -> None:
    store = InMemoryDocumentStore()
    authority, fence, permit = _authority(store)
    repository = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=authority,
    )

    manifest = _manifest(
        sequence=1,
        delivery_id="a" * 64,
        document_id="document-a",
    )
    assert repository.commit(
        manifest,
        expected_fence=fence,
        publication_permit=permit,
    ) is ManifestCommitStatus.COMMITTED
    descriptor = authority.read_committed_publication(
        tenant_id=_TENANT,
        binding_id=_BINDING,
    )
    assert descriptor is not None
    assert descriptor.manifest_id == manifest.manifest_id
    assert descriptor.manifest_fingerprint == manifest.manifest_fingerprint
    assert repository.get_committed_for_delivery(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        delivery_id=manifest.delivery_id,
    ) == manifest


def test_publication_chain_preserves_all_committed_pages() -> None:
    store = InMemoryDocumentStore()
    authority, fence, permit = _authority(store)
    repository = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=authority,
    )
    manifests = tuple(
        _manifest(
            sequence=sequence,
            delivery_id=chr(96 + sequence) * 64,
            document_id=f"document-{sequence}",
        )
        for sequence in (1, 2, 3)
    )
    for manifest in manifests:
        assert repository.commit(
            manifest,
            expected_fence=fence,
            publication_permit=permit,
        ) is ManifestCommitStatus.COMMITTED

    descriptors = authority.list_committed_publications(
        tenant_id=_TENANT,
        binding_id=_BINDING,
    )
    assert [item.materialization_sequence for item in descriptors] == [3, 2, 1]
    assert descriptors[0].previous_publication_commit_id == (
        descriptors[1].publication_commit_id
    )
    assert descriptors[1].previous_publication_commit_id == (
        descriptors[2].publication_commit_id
    )
    assert descriptors[2].previous_publication_commit_id is None
    assert repository.list_committed(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
    ) == manifests


def test_crash_after_fence_cas_keeps_history_for_later_page() -> None:
    class _CrashAfterFenceCASStore(InMemoryDocumentStore):
        crash_next_fence_cas = False

        def replace_if_match(self, *, expected, replacement):
            replaced = super().replace_if_match(
                expected=expected,
                replacement=replacement,
            )
            if (
                replaced
                and self.crash_next_fence_cas
                and expected.row_key == f"binding:{_BINDING}"
                and replacement.data.get("committed_publication") is not None
            ):
                self.crash_next_fence_cas = False
                raise RuntimeError("injected crash after fence CAS")
            return replaced

    store = _CrashAfterFenceCASStore()
    authority, fence, permit = _authority(store)
    repository = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=authority,
    )
    first = _manifest(sequence=1, delivery_id="a" * 64, document_id="document-a")
    second = _manifest(sequence=2, delivery_id="b" * 64, document_id="document-b")
    third = _manifest(sequence=3, delivery_id="c" * 64, document_id="document-c")
    assert repository.commit(
        first,
        expected_fence=fence,
        publication_permit=permit,
    ) is ManifestCommitStatus.COMMITTED
    store.crash_next_fence_cas = True
    with pytest.raises(RuntimeError, match="injected crash"):
        repository.commit(
            second,
            expected_fence=fence,
            publication_permit=permit,
        )
    assert authority.release_publication_permit(permit) is True
    next_permit = authority.acquire_publication_permit(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        expected_revision=1,
        expected_token=_TOKEN,
        owner_id="owner-restart",
        ttl_seconds=60,
    )
    assert next_permit is not None
    assert repository.commit(
        third,
        expected_fence=fence,
        publication_permit=next_permit,
    ) is ManifestCommitStatus.COMMITTED

    assert repository.list_committed(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_INDEXED,
        knowledge_source_binding_ref=_BINDING,
    ) == (first, second, third)


def test_remote_candidate_index_preserves_unrelated_pages_and_supersedes_item() -> None:
    store = InMemoryDocumentStore()
    authority, fence, permit = _authority(store)
    repository = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=authority,
    )
    page_one = _manifest(
        sequence=1,
        delivery_id="a" * 64,
        document_id="document-a",
        entries=(
            _entry("remote-a", "document-a"),
            _entry("remote-b", "document-b"),
        ),
    )
    page_two = _manifest(
        sequence=2,
        delivery_id="b" * 64,
        document_id="document-c",
        remote_id="remote-c",
    )
    page_three = _manifest(
        sequence=3,
        delivery_id="c" * 64,
        document_id="document-a-v2",
        remote_id="remote-a",
    )
    for manifest in (page_one, page_two, page_three):
        assert repository.commit(
            manifest,
            expected_fence=fence,
            publication_permit=permit,
        ) is ManifestCommitStatus.COMMITTED

    def remote(remote_id: str):
        return repository.get_committed_for_remote(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_INDEXED,
            knowledge_source_binding_ref=_BINDING,
            remote_id=remote_id,
        )

    assert remote("remote-a") == page_three
    assert remote("remote-b") == page_one
    assert remote("remote-c") == page_two


def test_manifest_commit_rejects_permit_expiry_before_authority_cas() -> None:
    store = InMemoryDocumentStore()
    now = {"value": _NOW}
    authority = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        store,
        clock=lambda: now["value"],
        permit_id_factory=lambda: "permit-expiry",
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
    permit = authority.acquire_publication_permit(
        tenant_id=_TENANT,
        binding_id=_BINDING,
        expected_revision=1,
        expected_token=_TOKEN,
        owner_id="owner-a",
        ttl_seconds=1,
    )
    assert permit is not None
    now["value"] = _NOW.replace(second=_NOW.second + 2)
    repository = ConnectedSourceMaterializationManifestRepository(
        store,
        publication_authority=authority,
    )
    with pytest.raises(RuntimeError, match="expired"):
        repository.commit(
            _manifest(
                sequence=1,
                delivery_id="a" * 64,
                document_id="document-a",
            ),
            expected_fence=fence,
            publication_permit=permit,
        )
    assert authority.read_committed_publication(
        tenant_id=_TENANT,
        binding_id=_BINDING,
    ) is None
