# © Artur Czarnecki. All rights reserved.

"""MP-3F — artifact content storage boundary tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping, Optional

import pytest
from pydantic import ValidationError

from intergrax.collaborative_work.artifact_service import (
    CollaborativeWorkArtifactService,
    TRUSTED_OPERATION_WORK_ARTIFACT_CREATE,
    TRUSTED_OPERATION_WORK_ARTIFACT_PUBLISH,
)
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.content_storage import (
    ArtifactContentIntegrityError,
    ArtifactContentPersistenceError,
    ArtifactContentReferenceUnsupported,
    ArtifactContentStore,
    GetArtifactContentRequest,
    ObjectStorageArtifactContentStore,
    StoreArtifactContentRequest,
    StoredArtifactContent,
)
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkItemRepository,
    InMemoryWorkspaceMembershipRepository,
    open_in_memory_artifact_repositories,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CreateCollaborativeOperationPolicyProfileCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkItemCommand,
    CreateWorkspaceMembershipCommand,
    WorkArtifactVersionRepository,
)
from intergrax.contracts.collaborative_work import (
    ArtifactContentRef,
    CollaborativeOperationPolicyProfileStatus,
    CreateWorkArtifactRequest,
    MembershipResolutionMode,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
    WorkspaceMembershipRole,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.validation import compute_sha256_content_digest
from intergrax.integrations.contracts.object_storage import ObjectStorage, StoredObject

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE_A = "workspace-a"
_WORKSPACE_B = "workspace-b"
_ACTING = "principal-acting"
_AUTHORITY_SCOPE = "collaborative_work.manage"
_WORK_ITEM_ID = "work-item-1"
_ARTIFACT_ID = "artifact-1"
_VERSION_ID = "artifact-version-1"
_NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
_SAMPLE_BODY = b'{"hello":"artifact"}'
_ALT_BODY = b"plain-text-body"


class _UnusedRuntimeEvaluator:
    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        _ = request
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="runtime evaluator must not run for internal artifact mutations",
            policy_rule_id="test.unexpected_runtime",
        )


class InMemoryObjectStorage:
    """Strict in-memory ``ObjectStorage`` test double."""

    def __init__(self) -> None:
        self._objects: dict[str, StoredObject] = {}

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._objects[key] = StoredObject(
            key=key,
            body=body,
            content_type=content_type,
            metadata=dict(metadata or {}),
            size_bytes=len(body),
        )

    def get(self, key: str) -> StoredObject | None:
        return self._objects.get(key)

    def delete(self, key: str) -> None:
        self._objects.pop(key, None)

    def presigned_url(self, key: str, *, expires_in_seconds: int = 3600, method: str = "GET") -> str:
        return f"https://example.test/{key}?method={method}&exp={expires_in_seconds}"

    def close(self) -> None:
        return None

    def corrupt(self, key: str, body: bytes) -> None:
        existing = self._objects.get(key)
        if existing is None:
            raise KeyError(key)
        self._objects[key] = StoredObject(
            key=key,
            body=body,
            content_type=existing.content_type,
            metadata=existing.metadata,
            size_bytes=len(body),
        )


def _store_request(
    *,
    tenant_id: str = _TENANT_A,
    workspace_id: str = _WORKSPACE_A,
    body: bytes = _SAMPLE_BODY,
    media_type: str = "application/json",
) -> StoreArtifactContentRequest:
    return StoreArtifactContentRequest(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        body=body,
        media_type=media_type,
    )


def _get_request(
    content_ref: ArtifactContentRef,
    *,
    tenant_id: str = _TENANT_A,
    workspace_id: str = _WORKSPACE_A,
) -> GetArtifactContentRequest:
    return GetArtifactContentRequest(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        content_ref=content_ref,
    )


def _physical_key_for(
    store: ObjectStorageArtifactContentStore,
    request: StoreArtifactContentRequest,
    content_ref: ArtifactContentRef,
) -> str:
    from intergrax.collaborative_work.content_storage import _physical_object_key

    digest_hex = content_ref.integrity_digest.removeprefix("sha256:")
    return _physical_object_key(request.tenant_id, request.workspace_id, digest_hex)


def _profile_command(operation_id: str) -> CreateCollaborativeOperationPolicyProfileCommand:
    return CreateCollaborativeOperationPolicyProfileCommand(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        operation_id=operation_id,
        authority_scope=_AUTHORITY_SCOPE,
        workspace_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
        resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
        runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
        resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
        meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
        status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
    )


@dataclass(frozen=True, slots=True)
class _PublicationFixture:
    artifact_service: CollaborativeWorkArtifactService
    content_store: ObjectStorageArtifactContentStore
    version_repo: WorkArtifactVersionRepository


def _publication_fixture() -> _PublicationFixture:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    delegation_repo = InMemoryAuthorityDelegationRepository()
    work_item_repo = InMemoryWorkItemRepository()
    artifact_bundle = open_in_memory_artifact_repositories()
    object_storage = InMemoryObjectStorage()
    content_store = ObjectStorageArtifactContentStore(object_storage)

    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            membership_id="membership-acting",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            authority_grant_id="authority-grant-acting",
            principal_id=_ACTING,
            authority_scopes=(_AUTHORITY_SCOPE,),
        )
    )
    for operation_id in (
        TRUSTED_OPERATION_WORK_ARTIFACT_CREATE,
        TRUSTED_OPERATION_WORK_ARTIFACT_PUBLISH,
    ):
        profile_repo.create(_profile_command(operation_id))
    work_item_repo.create(
        CreateWorkItemCommand(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_item_id=_WORK_ITEM_ID,
            created_by_principal_id=_ACTING,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )

    gate = CollaborativeWorkEnforcementGate(
        profile_repository=profile_repo,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=delegation_repo,
            principal_authority_repository=authority_repo,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(policy_repo),
        runtime_policy_evaluator=_UnusedRuntimeEvaluator(),
    )
    artifact_service = CollaborativeWorkArtifactService(
        work_item_repository=work_item_repo,
        work_artifact_repository=artifact_bundle.artifact,
        artifact_publication_repository=artifact_bundle.publication,
        enforcement_gate=gate,
        clock=lambda: _NOW,
    )
    return _PublicationFixture(
        artifact_service=artifact_service,
        content_store=content_store,
        version_repo=artifact_bundle.version,
    )


# --- CONTRACT ---


def test_store_request_is_strict() -> None:
    request = _store_request()
    assert request.body == _SAMPLE_BODY
    with pytest.raises(ValidationError):
        StoreArtifactContentRequest(
            tenant_id="",
            workspace_id=_WORKSPACE_A,
            body=b"x",
            media_type="text/plain",
        )
    with pytest.raises(ValidationError):
        StoreArtifactContentRequest(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            body=b"x",
            media_type="text/plain",
            digest="sha256:" + ("a" * 64),
        )


def test_store_request_allows_empty_body() -> None:
    request = StoreArtifactContentRequest(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        body=b"",
        media_type="application/octet-stream",
    )
    assert request.body == b""


def test_get_request_is_strict() -> None:
    content_ref = ObjectStorageArtifactContentStore(InMemoryObjectStorage()).put(_store_request())
    request = _get_request(content_ref)
    assert request.content_ref == content_ref
    with pytest.raises(ValidationError):
        GetArtifactContentRequest(
            tenant_id="",
            workspace_id=_WORKSPACE_A,
            content_ref=content_ref,
        )


def test_stored_result_is_strict() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    content_ref = store.put(_store_request())
    stored = store.get(_get_request(content_ref))
    assert stored is not None
    assert isinstance(stored, StoredArtifactContent)
    with pytest.raises(ValidationError):
        StoredArtifactContent(content_ref=content_ref, body=_SAMPLE_BODY, extra="nope")


def test_artifact_content_store_runtime_protocol() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    assert isinstance(store, ArtifactContentStore)


def test_object_storage_runtime_protocol() -> None:
    assert isinstance(InMemoryObjectStorage(), ObjectStorage)


# --- PUT ---


def test_put_derives_canonical_digest_and_logical_ref() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    content_ref = store.put(_store_request())
    expected_digest = compute_sha256_content_digest(_SAMPLE_BODY)
    assert content_ref.integrity_digest == expected_digest
    assert content_ref.content_ref == f"artifact-content://sha256/{expected_digest.removeprefix('sha256:')}"


def test_put_sets_exact_size_and_media_type() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    content_ref = store.put(_store_request(media_type="text/plain", body=_ALT_BODY))
    assert content_ref.size_bytes == len(_ALT_BODY)
    assert content_ref.media_type == "text/plain"


def test_put_same_bytes_same_ref() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    first = store.put(_store_request())
    second = store.put(_store_request())
    assert first == second


def test_put_does_not_overwrite_existing_valid_content() -> None:
    backend = InMemoryObjectStorage()
    store = ObjectStorageArtifactContentStore(backend)
    request = _store_request()
    first = store.put(request)
    key = _physical_key_for(store, request, first)
    original = backend.get(key)
    assert original is not None
    store.put(request)
    after = backend.get(key)
    assert after is not None
    assert after.body == original.body


# --- GET ---


def test_get_round_trip() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    content_ref = store.put(_store_request())
    stored = store.get(_get_request(content_ref))
    assert stored is not None
    assert stored.body == _SAMPLE_BODY


def test_get_missing_returns_none() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    content_ref = ObjectStorageArtifactContentStore(InMemoryObjectStorage()).put(_store_request())
    assert store.get(_get_request(content_ref)) is None


def test_get_wrong_tenant_returns_none() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    content_ref = store.put(_store_request())
    assert store.get(_get_request(content_ref, tenant_id=_TENANT_B)) is None


def test_get_wrong_workspace_returns_none() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    content_ref = store.put(_store_request())
    assert store.get(_get_request(content_ref, workspace_id=_WORKSPACE_B)) is None


def test_get_wrong_logical_digest_vs_integrity_digest_fails() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    content_ref = store.put(_store_request())
    mismatched = ArtifactContentRef(
        content_ref=content_ref.content_ref,
        media_type=content_ref.media_type,
        integrity_digest="sha256:" + ("b" * 64),
        size_bytes=content_ref.size_bytes,
    )
    with pytest.raises(ArtifactContentIntegrityError):
        store.get(_get_request(mismatched))


def test_get_body_corruption_fails() -> None:
    backend = InMemoryObjectStorage()
    store = ObjectStorageArtifactContentStore(backend)
    request = _store_request()
    content_ref = store.put(request)
    key = _physical_key_for(store, request, content_ref)
    backend.corrupt(key, b"corrupted")
    with pytest.raises(ArtifactContentIntegrityError):
        store.get(_get_request(content_ref))


def test_get_wrong_size_fails() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    content_ref = store.put(_store_request())
    wrong_size = ArtifactContentRef(
        content_ref=content_ref.content_ref,
        media_type=content_ref.media_type,
        integrity_digest=content_ref.integrity_digest,
        size_bytes=(content_ref.size_bytes or 0) + 1,
    )
    with pytest.raises(ArtifactContentIntegrityError):
        store.get(_get_request(wrong_size))


def test_get_size_none_digest_only_success() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    content_ref = store.put(_store_request())
    digest_only = ArtifactContentRef(
        content_ref=content_ref.content_ref,
        media_type=content_ref.media_type,
        integrity_digest=content_ref.integrity_digest,
        size_bytes=None,
    )
    stored = store.get(_get_request(digest_only))
    assert stored is not None
    assert stored.body == _SAMPLE_BODY


# --- IMMUTABILITY ---


def test_corrupted_existing_object_not_repaired_by_put() -> None:
    backend = InMemoryObjectStorage()
    store = ObjectStorageArtifactContentStore(backend)
    request = _store_request()
    content_ref = store.put(request)
    key = _physical_key_for(store, request, content_ref)
    backend.corrupt(key, b"corrupted")
    with pytest.raises(ArtifactContentIntegrityError):
        store.put(request)


def test_put_is_naturally_idempotent() -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    refs = [store.put(_store_request()) for _ in range(3)]
    assert refs[0] == refs[1] == refs[2]


# --- PROVIDER NEUTRALITY ---


def test_two_object_storage_instances_same_logical_ref() -> None:
    first = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    second = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    request = _store_request()
    ref_a = first.put(request)
    ref_b = second.put(request)
    assert ref_a.content_ref == ref_b.content_ref
    assert ref_a.integrity_digest == ref_b.integrity_digest
    assert ref_a.size_bytes == ref_b.size_bytes


def test_artifact_content_ref_has_no_provider_identity() -> None:
    content_ref = ObjectStorageArtifactContentStore(InMemoryObjectStorage()).put(_store_request())
    serialized = content_ref.model_dump_json()
    lowered = serialized.lower()
    assert "s3" not in lowered
    assert "gcs" not in lowered
    assert "azure" not in lowered
    assert "filesystem" not in lowered
    assert "minio" not in lowered
    assert content_ref.content_ref.startswith("artifact-content://sha256/")


def test_artifact_content_ref_has_no_physical_key() -> None:
    backend = InMemoryObjectStorage()
    store = ObjectStorageArtifactContentStore(backend)
    request = _store_request()
    content_ref = store.put(request)
    key = _physical_key_for(store, request, content_ref)
    assert key not in content_ref.content_ref
    assert key not in content_ref.model_dump_json()


# --- CONTENT REF FORMAT ---


@pytest.mark.parametrize(
    "invalid_ref",
    (
        "artifact-content://sha256/" + ("A" * 64),
        "artifact-content://md5/" + ("a" * 32),
        "artifact-content://sha256/" + ("a" * 63),
        "artifact-content://sha256/" + ("a" * 64) + "/extra",
        "artifact-content://sha256/" + ("a" * 64) + "?q=1",
        "artifact-content://sha256/" + ("a" * 64) + "#frag",
        "s3://bucket/key",
        "gs://bucket/key",
        "file:///tmp/blob",
    ),
)
def test_unsupported_logical_content_ref_formats_fail(invalid_ref: str) -> None:
    store = ObjectStorageArtifactContentStore(InMemoryObjectStorage())
    digest = compute_sha256_content_digest(_SAMPLE_BODY)
    content_ref = ArtifactContentRef(
        content_ref=invalid_ref,
        media_type="application/json",
        integrity_digest=digest,
        size_bytes=len(_SAMPLE_BODY),
    )
    with pytest.raises((ArtifactContentReferenceUnsupported, ArtifactContentIntegrityError, ValidationError)):
        store.get(_get_request(content_ref))


# --- MP-3 INTEGRATION ---


def test_mp3_publication_round_trip_without_raw_body_in_metadata() -> None:
    fixture = _publication_fixture()
    content_ref = fixture.content_store.put(_store_request())
    published = fixture.artifact_service.create_artifact(
        CreateWorkArtifactRequest(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_item_id=_WORK_ITEM_ID,
            work_artifact_id=_ARTIFACT_ID,
            work_artifact_version_id=_VERSION_ID,
            acting_principal_id=_ACTING,
            content_ref=content_ref,
            idempotency_key="artifact-create-1",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        )
    )
    stored_version = fixture.version_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_artifact_version_id=_VERSION_ID,
    )
    assert stored_version is not None
    assert stored_version.content_ref == content_ref
    retrieved = fixture.content_store.get(_get_request(stored_version.content_ref))
    assert retrieved is not None
    assert retrieved.body == _SAMPLE_BODY

    version_json = json.loads(stored_version.model_dump_json())
    serialized = json.dumps(version_json)
    assert _SAMPLE_BODY.decode("utf-8") not in serialized
    assert "s3://" not in serialized
    assert published.version.content_ref == content_ref


def test_put_persistence_failure_when_unreadable_after_write() -> None:
    class _UnreadableAfterPut(InMemoryObjectStorage):
        def put(
            self,
            key: str,
            body: bytes,
            *,
            content_type: str = "application/octet-stream",
            metadata: Optional[Mapping[str, str]] = None,
        ) -> None:
            super().put(key, body, content_type=content_type, metadata=metadata)
            self._objects.pop(key, None)

    store = ObjectStorageArtifactContentStore(_UnreadableAfterPut())
    with pytest.raises(ArtifactContentPersistenceError):
        store.put(_store_request())
