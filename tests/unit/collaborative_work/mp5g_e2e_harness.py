# © Artur Czarnecki. All rights reserved.

"""MP-5G orchestration harness — principals, seeds, authority→policy→composer wiring only."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypeVar

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.context_view_composition import DefaultContextViewComposer
from intergrax.collaborative_work.context_view_source_adapters import (
    DefaultCollaborativeWorkContextSource,
    DefaultKnowledgeContextSource,
    DefaultMemoryContextSource,
    DefaultUclContextSource,
)
from intergrax.collaborative_work.context_view_visibility import (
    ContextViewVisibilityEvaluator,
    DefaultContextViewVisibilityPolicy,
)
from intergrax.collaborative_work.contracts.collaborative_work_reference_read import (
    CollaborativeWorkReferenceReadPort,
    CollaborativeWorkReferenceReadRequest,
    CollaborativeWorkReferenceReadResult,
)
from intergrax.collaborative_work.default_collaborative_work_reference_reader import (
    CollaborativeWorkReferenceReadCapabilityBinding,
    DefaultCollaborativeWorkReferenceReader,
)
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkItemRepository,
    InMemoryWorkspaceMembershipRepository,
    open_in_memory_artifact_repositories,
)
from intergrax.collaborative_work.repository import (
    CreateArtifactWithInitialVersionCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkItemCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.collaborative_work.repository_backed_reference_catalog import (
    RepositoryBackedCollaborativeWorkReferenceCatalog,
)
from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.contracts.collaborative_work import (
    ArtifactContentRef,
    MembershipResolutionMode,
    WorkItemState,
    WorkspaceMembershipRole,
)
from intergrax.contracts.context_view import (
    ContextView,
    ContextViewCategory,
    ContextViewRequest,
    ContextViewScope,
)
from intergrax.contracts.context_view_composition import (
    ContextViewCompositionPolicyDeniedError,
    ContextViewCompositionRequest,
    ContextViewCompositionRequestAlignmentError,
    ContextViewCompositionSourceFailureError,
    DefaultContextViewComposerConfig,
)
from intergrax.contracts.context_view_scope_compatibility import ContextViewScopeCompatibilityPolicy
from intergrax.contracts.context_view_source_ports import (
    CollaborativeWorkContextSourcePort,
    ContextViewCollaborativeWorkSourceCandidatesResult,
    ContextViewCollaborativeWorkSourceRequest,
    ContextViewKnowledgeSourceCandidatesResult,
    ContextViewKnowledgeSourceRequest,
    ContextViewMemorySourceCandidatesResult,
    ContextViewMemorySourceRequest,
    ContextViewUclSourceCandidatesResult,
    ContextViewUclSourceRequest,
    KnowledgeContextSourcePort,
    MemoryContextSourcePort,
    UclContextSourcePort,
)
from intergrax.contracts.context_view_visibility_policy import (
    CONTEXT_VIEW_READ_AUTHORITY_SCOPE,
    ContextViewPolicyDecision,
    ContextViewPolicyOutcome,
)
from intergrax.knowledge.contracts.knowledge_reference_read import (
    KnowledgeReferenceReadPort,
    KnowledgeReferenceReadRequest,
    KnowledgeReferenceReadResult,
)
from intergrax.memory.contracts.memory_reference_read import (
    MemoryReferenceReadPort,
    MemoryReferenceReadRequest,
    MemoryReferenceReadResult,
)
from intergrax.memory.default_memory_control_plane import UserProfileManagerMemoryCapability
from intergrax.memory.default_memory_reference_reader import (
    DefaultMemoryReferenceReader,
    MemoryReferenceReadCapabilityBinding,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry
from intergrax.rag.default_knowledge_reference_reader import (
    DefaultKnowledgeReferenceReader,
    KnowledgeReferenceReadCapabilityBinding,
)
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk, RetrievalResult, RetrievalTrace
from intergrax.runtime.context_lifecycle import (
    ArtifactCompressionTarget,
    ArtifactLookupKey,
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    InMemoryOptimizationArtifactRepository,
    OptimizationArtifactRepository,
    OptimizationArtifactScopedReferenceCatalog,
    OptimizationArtifactType,
    ReusableArtifactStatus,
    ReusableOptimizationArtifact,
    StoredOptimizationArtifact,
    UclArtifactOwnership,
    UclArtifactOwnershipKind,
    UclArtifactOwnershipScope,
    compute_artifact_content_hash,
)
from intergrax.runtime.context_lifecycle.default_ucl_reference_reader import (
    DefaultUclReferenceReader,
    UclReferenceReadCapabilityBinding,
)
from intergrax.ucl.contracts.ucl_reference_read import (
    UclReferenceReadPort,
    UclReferenceReadRequest,
    UclReferenceReadResult,
)

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"
WS_A = "workspace-a"
WS_B = "workspace-b"
WS_X = "workspace-x"
WI_A1 = "work-item-a1"
WI_A2 = "work-item-a2"
WI_B1 = "work-item-b1"
WI_X1 = "work-item-x1"
CTX_A = "context-a"
CTX_B = "context-b"
CTX_X = "context-x"
DOC_A1 = "document-a1"
DOC_A2 = "document-a2"
DOC_B1 = "document-b1"
DOC_X1 = "document-x1"
PRINCIPAL_A = "principal-a"
PRINCIPAL_B = "principal-b"
PRINCIPAL_C = "principal-c"
SERVICE_PRINCIPAL = "service-runtime-a"
OP_COMPOSE = "collaborative_work.context_view.compose"
KNOWLEDGE_QUERY = "mp5g-qualification-enumerate"
_NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
_DIGEST = "sha256:" + ("a" * 64)

T = TypeVar("T")


class ImmediateAsyncRunner:
    def run(self, coro):  # type: ignore[no-untyped-def]
        return asyncio.run(coro)


class WorkspaceRoutedMemoryReader:
    """Routes to per-workspace DefaultMemoryReferenceReader instances."""

    def __init__(self, readers: dict[str, DefaultMemoryReferenceReader]) -> None:
        self._readers = readers
        self.calls: list[tuple[RequestIdentity, MemoryReferenceReadRequest]] = []

    async def read_references(
        self,
        identity: RequestIdentity,
        request: MemoryReferenceReadRequest,
    ) -> MemoryReferenceReadResult:
        self.calls.append((identity, request))
        reader = self._readers.get(request.scope.workspace_id)
        if reader is None:
            from intergrax.memory.contracts.memory_reference_read import MemoryReferenceReadOutcome

            return MemoryReferenceReadResult(
                outcome=MemoryReferenceReadOutcome.UNAVAILABLE,
                reason="no_reader_for_workspace",
            )
        return await reader.read_references(identity, request)


class _KnowledgeBackend:
    def __init__(self, chunks: tuple[RetrievalChunk, ...]) -> None:
        self._chunks = chunks
        self.last_request: RetrievalRequest | None = None

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.last_request = request
        return RetrievalResult(
            chunks=list(self._chunks),
            used=True,
            reason="ok",
            trace=RetrievalTrace(),
        )


class WorkspaceRoutedKnowledgeReader:
    def __init__(self, readers: dict[str, DefaultKnowledgeReferenceReader]) -> None:
        self._readers = readers
        self.calls: list[tuple[RequestIdentity, KnowledgeReferenceReadRequest]] = []

    def read_references(
        self,
        identity: RequestIdentity,
        request: KnowledgeReferenceReadRequest,
    ) -> KnowledgeReferenceReadResult:
        self.calls.append((identity, request))
        reader = self._readers.get(request.scope.workspace_id)
        if reader is None:
            from intergrax.knowledge.contracts.knowledge_reference_read import (
                KnowledgeReferenceReadOutcome,
            )

            return KnowledgeReferenceReadResult(
                outcome=KnowledgeReferenceReadOutcome.UNAVAILABLE,
            )
        return reader.read_references(identity, request)


class WorkspaceRoutedCollaborativeWorkReader:
    def __init__(
        self,
        readers: dict[str, DefaultCollaborativeWorkReferenceReader],
    ) -> None:
        self._readers = readers
        self.calls: list[tuple[RequestIdentity, CollaborativeWorkReferenceReadRequest]] = []

    def read_references(
        self,
        identity: RequestIdentity,
        request: CollaborativeWorkReferenceReadRequest,
    ) -> CollaborativeWorkReferenceReadResult:
        self.calls.append((identity, request))
        reader = self._readers.get(request.scope.workspace_id)
        if reader is None:
            from intergrax.collaborative_work.contracts.collaborative_work_reference_read import (
                CollaborativeWorkReferenceReadOutcome,
            )

            return CollaborativeWorkReferenceReadResult(
                outcome=CollaborativeWorkReferenceReadOutcome.UNAVAILABLE,
            )
        return reader.read_references(identity, request)


class WorkspaceRoutedUclReader:
    def __init__(self, readers: dict[str, DefaultUclReferenceReader]) -> None:
        self._readers = readers
        self.calls: list[tuple[RequestIdentity, UclReferenceReadRequest]] = []

    async def read_references(
        self,
        identity: RequestIdentity,
        request: UclReferenceReadRequest,
    ) -> UclReferenceReadResult:
        self.calls.append((identity, request))
        reader = self._readers.get(request.scope.workspace_id)
        if reader is None:
            from intergrax.ucl.contracts.ucl_reference_read import UclReferenceReadOutcome

            return UclReferenceReadResult(outcome=UclReferenceReadOutcome.UNAVAILABLE)
        return await reader.read_references(identity, request)


def _ucl_lookup_key(
    *,
    tenant_id: str = TENANT_A,
    context_scope_id: str = CTX_A,
    source_content_hash: str = "hash-abc",
    source_refs: tuple[str, ...] = ("msg-1", "msg-2"),
) -> ArtifactLookupKey:
    return ArtifactLookupKey(
        tenant_id=tenant_id,
        context_scope_id=context_scope_id,
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash=source_content_hash,
        strategy_id="strategy.summarize",
        strategy_version="1.0.0",
        policy_version="policy-v1",
        validation_contract_version="validation-v1",
        compression_target=ArtifactCompressionTarget(target_tokens=1000),
        lossiness_profile="lossy_summary",
        source_refs=source_refs,
    )


def _ucl_stored_artifact(
    *,
    artifact_id: str,
    workspace_id: str,
    context_scope_id: str,
    tenant_id: str = TENANT_A,
) -> StoredOptimizationArtifact:
    key = _ucl_lookup_key(tenant_id=tenant_id, context_scope_id=context_scope_id)
    ownership = UclArtifactOwnership.for_workspace(
        UclArtifactOwnershipScope(tenant_id=tenant_id, workspace_id=workspace_id),
    )
    payload = f"payload-{artifact_id}".encode()
    metadata = ReusableOptimizationArtifact(
        artifact_id=artifact_id,
        lookup_key=key,
        ownership=ownership,
        artifact_content_hash=compute_artifact_content_hash(payload),
        created_at=_NOW,
        created_by_executor="executor.message_sequence",
        validation=ArtifactValidationSummary(
            status=ArtifactValidationStatus.PASSED,
            validation_contract_version="validation-v1",
            validated_at=_NOW,
        ),
        status=ReusableArtifactStatus.VALIDATED,
    )
    return StoredOptimizationArtifact(
        metadata=metadata,
        payload=payload,
        media_type="application/octet-stream",
    )


def _ucl_publish_artifact(
    repository: OptimizationArtifactRepository,
    artifact: StoredOptimizationArtifact,
) -> None:
    key = artifact.metadata.lookup_key
    scope = artifact.metadata.ownership.scope
    if scope is None:
        raise AssertionError("workspace publish requires WORKSPACE ownership")
    reservation = repository.try_acquire_creation_reservation(
        key,
        ownership=scope,
        owner_operation_id="op-mp5g",
        lease_seconds=60,
    )
    assert reservation.reservation is not None
    repository.store_validated_artifact(
        reservation=reservation.reservation,
        artifact=artifact,
    )


def _knowledge_chunk(
    *,
    tenant_id: str,
    workspace_id: str,
    vector_id: str,
    document_id: str,
) -> RetrievalChunk:
    return RetrievalChunk(
        id=document_id,
        text="forbidden-body",
        score=0.9,
        rank=1,
        channel="dense",
        vector_id=vector_id,
        scope={"tenant_id": tenant_id, "workspace_id": workspace_id, "namespace": None},
        provenance={
            "source_id": f"src-{document_id}",
            "source_kind": "file",
            "root_document_id": document_id,
        },
    )


def _cw_content_ref(tenant_id: str, workspace_id: str) -> ArtifactContentRef:
    return ArtifactContentRef.model_validate(
        {
            "content_ref": f"content://{tenant_id}/{workspace_id}/body",
            "media_type": "application/json",
            "integrity_digest": _DIGEST,
        }
    )


@dataclass
class RecordingMemoryPort(MemoryContextSourcePort):
    inner: DefaultMemoryContextSource
    calls: list[ContextViewMemorySourceRequest] = field(default_factory=list)

    def list_candidates(
        self,
        request: ContextViewMemorySourceRequest,
    ) -> ContextViewMemorySourceCandidatesResult:
        self.calls.append(request)
        return self.inner.list_candidates(request)


@dataclass
class RecordingKnowledgePort(KnowledgeContextSourcePort):
    inner: DefaultKnowledgeContextSource
    calls: list[ContextViewKnowledgeSourceRequest] = field(default_factory=list)

    def list_candidates(
        self,
        request: ContextViewKnowledgeSourceRequest,
    ) -> ContextViewKnowledgeSourceCandidatesResult:
        self.calls.append(request)
        return self.inner.list_candidates(request)


@dataclass
class RecordingUclPort(UclContextSourcePort):
    inner: DefaultUclContextSource
    calls: list[ContextViewUclSourceRequest] = field(default_factory=list)

    def list_candidates(
        self,
        request: ContextViewUclSourceRequest,
    ) -> ContextViewUclSourceCandidatesResult:
        self.calls.append(request)
        return self.inner.list_candidates(request)


@dataclass
class RecordingCollaborativeWorkPort(CollaborativeWorkContextSourcePort):
    inner: DefaultCollaborativeWorkContextSource
    calls: list[ContextViewCollaborativeWorkSourceRequest] = field(default_factory=list)

    def list_candidates(
        self,
        request: ContextViewCollaborativeWorkSourceRequest,
    ) -> ContextViewCollaborativeWorkSourceCandidatesResult:
        self.calls.append(request)
        return self.inner.list_candidates(request)


@dataclass(frozen=True, slots=True)
class Mp5gHarness:
    evaluator: ContextViewVisibilityEvaluator
    composer: DefaultContextViewComposer
    memory_reader: WorkspaceRoutedMemoryReader
    knowledge_reader: WorkspaceRoutedKnowledgeReader
    ucl_reader: WorkspaceRoutedUclReader
    collaborative_work_reader: WorkspaceRoutedCollaborativeWorkReader
    collaborative_work_catalog: RepositoryBackedCollaborativeWorkReferenceCatalog
    memory_port: RecordingMemoryPort
    knowledge_port: RecordingKnowledgePort
    ucl_port: RecordingUclPort
    collaborative_work_port: RecordingCollaborativeWorkPort
    membership_repo: InMemoryWorkspaceMembershipRepository
    authority_repo: InMemoryPrincipalAuthorityRepository
    delegation_repo: InMemoryAuthorityDelegationRepository
    memory_entry_ids: dict[str, str]
    knowledge_refs: dict[str, str]


def principal_identity(
    *,
    tenant_id: str,
    principal_id: str,
    principal_type: PrincipalType = PrincipalType.USER,
) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=principal_id,
        principal_type=principal_type,
        auth_subject=principal_id,
    )


def context_view_scope(
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str = WS_A,
    work_item_id: str | None = None,
    **operation: object,
) -> ContextViewScope:
    from intergrax.contracts.context_view import ContextViewOperationScope

    operation_scope = None
    if operation:
        operation_scope = ContextViewOperationScope(**operation)  # type: ignore[arg-type]
    return ContextViewScope(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        work_item_id=work_item_id,
        operation_scope=operation_scope,
    )


def context_view_request(
    *,
    scope: ContextViewScope,
    acting_principal_id: str,
    categories: tuple[ContextViewCategory, ...],
    **overrides: object,
) -> ContextViewRequest:
    payload = {
        "scope": scope,
        "acting_principal_id": acting_principal_id,
        "operation_id": OP_COMPOSE,
        "requested_categories": categories,
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return ContextViewRequest(**payload)  # type: ignore[arg-type]


def _seed_authority_bundle(
    *,
    membership_repo: InMemoryWorkspaceMembershipRepository,
    authority_repo: InMemoryPrincipalAuthorityRepository,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
) -> None:
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            membership_id=f"mem-{tenant_id}-{workspace_id}-{principal_id}",
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            authority_grant_id=f"grant-{tenant_id}-{workspace_id}-{principal_id}",
            principal_id=principal_id,
            authority_scopes=(CONTEXT_VIEW_READ_AUTHORITY_SCOPE,),
        )
    )


async def _seed_memory(
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
    label: str,
) -> tuple[DefaultMemoryReferenceReader, str]:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store, tenant_id=tenant_id, workspace_id=workspace_id)
    capability = UserProfileManagerMemoryCapability(manager)
    entry = UserProfileMemoryEntry(content=f"memory-{label}", kind=MemoryKind.PREFERENCE)
    await capability.add_memory_entry(
        principal_identity(tenant_id=tenant_id, principal_id=principal_id),
        entry,
    )
    reader = DefaultMemoryReferenceReader(
        user_profile=capability,
        capability_binding=MemoryReferenceReadCapabilityBinding(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ),
    )
    return reader, entry.entry_id


def _seed_knowledge_reader(
    *,
    tenant_id: str,
    workspace_id: str,
    documents: tuple[tuple[str, str], ...],
) -> DefaultKnowledgeReferenceReader:
    chunks = tuple(
        _knowledge_chunk(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            vector_id=vector_id,
            document_id=document_id,
        )
        for document_id, vector_id in documents
    )
    backend = _KnowledgeBackend(chunks=chunks)
    return DefaultKnowledgeReferenceReader(
        retrieval=backend,
        capability_binding=KnowledgeReferenceReadCapabilityBinding(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ),
    )


def _seed_ucl_reader(
    *,
    tenant_id: str,
    workspace_id: str,
    context_scope_id: str,
    artifact_id: str,
) -> tuple[DefaultUclReferenceReader, InMemoryOptimizationArtifactRepository]:
    repo: OptimizationArtifactScopedReferenceCatalog = InMemoryOptimizationArtifactRepository()
    artifact = _ucl_stored_artifact(
        artifact_id=artifact_id,
        workspace_id=workspace_id,
        context_scope_id=context_scope_id,
        tenant_id=tenant_id,
    )
    _ucl_publish_artifact(repo, artifact)
    reader = DefaultUclReferenceReader(
        catalog=repo,
        capability_binding=UclReferenceReadCapabilityBinding(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            context_scope_id=context_scope_id,
        ),
    )
    return reader, repo


def build_mp5g_harness(
    *,
    scope_compatibility_policy: ContextViewScopeCompatibilityPolicy | None = None,
) -> Mp5gHarness:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    delegation_repo = InMemoryAuthorityDelegationRepository()

    for principal, tenant, workspace in (
        (PRINCIPAL_A, TENANT_A, WS_A),
        (PRINCIPAL_A, TENANT_A, WS_B),
        (PRINCIPAL_B, TENANT_A, WS_B),
        (PRINCIPAL_C, TENANT_B, WS_X),
        (SERVICE_PRINCIPAL, TENANT_A, WS_A),
    ):
        _seed_authority_bundle(
            membership_repo=membership_repo,
            authority_repo=authority_repo,
            tenant_id=tenant,
            workspace_id=workspace,
            principal_id=principal,
        )

    async def _mem_all() -> tuple[dict[str, DefaultMemoryReferenceReader], dict[str, str]]:
        readers: dict[str, DefaultMemoryReferenceReader] = {}
        ids: dict[str, str] = {}

        async def _workspace_reader(
            *,
            tenant_id: str,
            workspace_id: str,
            principal_id: str,
            labels: tuple[str, ...],
        ) -> tuple[DefaultMemoryReferenceReader, str]:
            store = InMemoryUserProfileStore()
            manager = UserProfileManager(store, tenant_id=tenant_id, workspace_id=workspace_id)
            capability = UserProfileManagerMemoryCapability(manager)
            first_id = ""
            for label in labels:
                entry = UserProfileMemoryEntry(
                    content=f"memory-{label}",
                    kind=MemoryKind.PREFERENCE,
                )
                await capability.add_memory_entry(
                    principal_identity(tenant_id=tenant_id, principal_id=principal_id),
                    entry,
                )
                if not first_id:
                    first_id = entry.entry_id
            reader = DefaultMemoryReferenceReader(
                user_profile=capability,
                capability_binding=MemoryReferenceReadCapabilityBinding(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                ),
            )
            return reader, first_id

        readers[WS_A], ids[WS_A] = await _workspace_reader(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            principal_id=PRINCIPAL_A,
            labels=("a1", "a2"),
        )
        readers[WS_B], ids[WS_B] = await _workspace_reader(
            tenant_id=TENANT_A,
            workspace_id=WS_B,
            principal_id=PRINCIPAL_B,
            labels=("b1",),
        )
        readers[WS_X], ids[WS_X] = await _workspace_reader(
            tenant_id=TENANT_B,
            workspace_id=WS_X,
            principal_id=PRINCIPAL_C,
            labels=("x1",),
        )
        return readers, ids

    memory_readers, memory_entry_ids = asyncio.run(_mem_all())

    knowledge_readers = {
        WS_A: _seed_knowledge_reader(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            documents=(
                (DOC_A1, "vec-a1"),
                (DOC_A2, "vec-a2"),
                (CTX_A, "vec-ctx-a"),
            ),
        ),
        WS_B: _seed_knowledge_reader(
            tenant_id=TENANT_A,
            workspace_id=WS_B,
            documents=((DOC_B1, "vec-b1"),),
        ),
        WS_X: _seed_knowledge_reader(
            tenant_id=TENANT_B,
            workspace_id=WS_X,
            documents=((DOC_X1, "vec-x1"),),
        ),
    }
    knowledge_refs = {DOC_A1: "vec-a1", DOC_B1: "vec-b1", DOC_X1: "vec-x1"}

    ucl_readers: dict[str, DefaultUclReferenceReader] = {}
    ucl_readers[WS_A], _ = _seed_ucl_reader(
        tenant_id=TENANT_A,
        workspace_id=WS_A,
        context_scope_id=CTX_A,
        artifact_id="ucl-art-a",
    )
    ucl_readers[WS_B], _ = _seed_ucl_reader(
        tenant_id=TENANT_A,
        workspace_id=WS_B,
        context_scope_id=CTX_B,
        artifact_id="ucl-art-b",
    )
    ucl_readers[WS_X], _ = _seed_ucl_reader(
        tenant_id=TENANT_B,
        workspace_id=WS_X,
        context_scope_id=CTX_X,
        artifact_id="ucl-art-x",
    )

    work_items = InMemoryWorkItemRepository()
    artifact_bundle = open_in_memory_artifact_repositories()
    for tenant, workspace, work_item_id in (
        (TENANT_A, WS_A, WI_A1),
        (TENANT_A, WS_A, WI_A2),
        (TENANT_A, WS_B, WI_B1),
        (TENANT_B, WS_X, WI_X1),
    ):
        work_items.create(
            CreateWorkItemCommand(
                tenant_id=tenant,
                workspace_id=workspace,
                work_item_id=work_item_id,
                created_by_principal_id=PRINCIPAL_A,
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
        artifact_bundle.publication.create_artifact_with_initial_version(
            CreateArtifactWithInitialVersionCommand(
                tenant_id=tenant,
                workspace_id=workspace,
                work_item_id=work_item_id,
                work_artifact_id=f"artifact-{work_item_id}",
                work_artifact_version_id=f"version-{work_item_id}",
                created_by_principal_id=PRINCIPAL_A,
                published_by_principal_id=PRINCIPAL_A,
                content_ref=_cw_content_ref(tenant, workspace),
                artifact_created_at=_NOW,
                artifact_updated_at=_NOW,
                version_created_at=_NOW,
                version_published_at=_NOW,
                execution=None,
            )
        )

    cw_catalog = RepositoryBackedCollaborativeWorkReferenceCatalog(
        work_item_repository=work_items,
        work_artifact_repository=artifact_bundle.artifact,
        work_artifact_version_repository=artifact_bundle.version,
    )
    cw_readers = {
        WS_A: DefaultCollaborativeWorkReferenceReader(
            catalog=cw_catalog,
            capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
                tenant_id=TENANT_A,
                workspace_id=WS_A,
            ),
        ),
        WS_B: DefaultCollaborativeWorkReferenceReader(
            catalog=cw_catalog,
            capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
                tenant_id=TENANT_A,
                workspace_id=WS_B,
            ),
        ),
        WS_X: DefaultCollaborativeWorkReferenceReader(
            catalog=cw_catalog,
            capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
                tenant_id=TENANT_B,
                workspace_id=WS_X,
            ),
        ),
    }
    cw_reader = WorkspaceRoutedCollaborativeWorkReader(cw_readers)

    memory_reader = WorkspaceRoutedMemoryReader(memory_readers)
    knowledge_reader = WorkspaceRoutedKnowledgeReader(knowledge_readers)
    ucl_reader = WorkspaceRoutedUclReader(ucl_readers)
    runner = ImmediateAsyncRunner()

    memory_port = RecordingMemoryPort(
        inner=DefaultMemoryContextSource(reader=memory_reader, async_runner=runner),
    )
    knowledge_port = RecordingKnowledgePort(
        inner=DefaultKnowledgeContextSource(reader=knowledge_reader),
    )
    ucl_port = RecordingUclPort(
        inner=DefaultUclContextSource(reader=ucl_reader, async_runner=runner),
    )
    collaborative_work_port = RecordingCollaborativeWorkPort(
        inner=DefaultCollaborativeWorkContextSource(reader=cw_reader),
    )

    composer = DefaultContextViewComposer(
        memory_source=memory_port,
        knowledge_source=knowledge_port,
        ucl_source=ucl_port,
        collaborative_work_source=collaborative_work_port,
        config=DefaultContextViewComposerConfig(
            knowledge_reference_read_query_text=KNOWLEDGE_QUERY,
        ),
        scope_compatibility_policy=scope_compatibility_policy,
    )

    evaluator = ContextViewVisibilityEvaluator(
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=delegation_repo,
            principal_authority_repository=authority_repo,
            clock=lambda: _NOW,
        ),
        visibility_policy=DefaultContextViewVisibilityPolicy(),
        delegation_repository=delegation_repo,
    )

    return Mp5gHarness(
        evaluator=evaluator,
        composer=composer,
        memory_reader=memory_reader,
        knowledge_reader=knowledge_reader,
        ucl_reader=ucl_reader,
        collaborative_work_reader=cw_reader,
        collaborative_work_catalog=cw_catalog,
        memory_port=memory_port,
        knowledge_port=knowledge_port,
        ucl_port=ucl_port,
        collaborative_work_port=collaborative_work_port,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
        memory_entry_ids=memory_entry_ids,
        knowledge_refs=knowledge_refs,
    )


def run_qualified_flow(
    harness: Mp5gHarness,
    *,
    request: ContextViewRequest,
    identity: RequestIdentity,
) -> tuple[ContextViewPolicyDecision, ContextView]:
    decision = harness.evaluator.evaluate(request)
    if decision.outcome is not ContextViewPolicyOutcome.ALLOW:
        return decision, None  # type: ignore[return-value]
    composition = ContextViewCompositionRequest(
        request=request,
        policy_decision=decision,
        principal_identity=identity,
    )
    view = harness.composer.compose(composition)
    return decision, view


def assert_no_payload_fields(entry_source_ref: object) -> None:
    forbidden = {
        "content",
        "body",
        "text",
        "payload",
        "embedding",
        "summary",
        "content_ref",
        "integrity_digest",
    }
    if dataclasses.is_dataclass(entry_source_ref):
        keys = {f.name for f in dataclasses.fields(entry_source_ref)}
    else:
        keys = set(type(entry_source_ref).model_fields.keys())
    assert forbidden.isdisjoint(keys)


