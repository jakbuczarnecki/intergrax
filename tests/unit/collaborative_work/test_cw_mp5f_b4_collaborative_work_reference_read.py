# © Artur Czarnecki. All rights reserved.

"""MP-5F-B4: Collaborative Work scoped reference-read contract, isolation, and pluginability."""

from __future__ import annotations

import ast
import dataclasses
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intergrax.collaborative_work.contracts.collaborative_work_reference_read import (
    COLLABORATIVE_WORK_REFERENCE_READ_MAX_LIMIT,
    CollaborativeWorkArtifactCanonicalRef,
    CollaborativeWorkItemCanonicalRef,
    CollaborativeWorkReferenceEntityKind,
    CollaborativeWorkReferenceReadOutcome,
    CollaborativeWorkReferenceReadPort,
    CollaborativeWorkReferenceReadQuery,
    CollaborativeWorkReferenceReadRequest,
    CollaborativeWorkReferenceReadResult,
    CollaborativeWorkReferenceReadScope,
    CollaborativeWorkReferenceReadScopeError,
    CollaborativeWorkVersionSelection,
    validate_collaborative_work_reference_read_request,
)
from intergrax.collaborative_work.default_collaborative_work_reference_reader import (
    CollaborativeWorkReferenceReadCapabilityBinding,
    DefaultCollaborativeWorkReferenceReader,
)
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryArtifactRepositories,
    InMemoryWorkItemRepository,
    open_in_memory_artifact_repositories,
)
from intergrax.collaborative_work.repository import (
    CreateArtifactWithInitialVersionCommand,
    CreateWorkItemCommand,
    PublishWorkArtifactVersionCommand,
    CollaborativeWorkScopedReferenceCatalog,
    CollaborativeWorkScopedReferenceListing,
    CollaborativeWorkScopedReferenceQuery,
)
from intergrax.collaborative_work.repository_backed_reference_catalog import (
    RepositoryBackedCollaborativeWorkReferenceCatalog,
)
from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.contracts.collaborative_work import (
    ArtifactContentRef,
    WorkArtifactVersionRef,
    WorkItemState,
)

pytestmark = pytest.mark.gate

_REPO = Path(__file__).resolve().parents[3]
_CONTRACT = (
    _REPO
    / "intergrax"
    / "collaborative_work"
    / "contracts"
    / "collaborative_work_reference_read.py"
)
_DEFAULT_READER = (
    _REPO
    / "intergrax"
    / "collaborative_work"
    / "default_collaborative_work_reference_reader.py"
)
_CATALOG = (
    _REPO
    / "intergrax"
    / "collaborative_work"
    / "repository_backed_reference_catalog.py"
)

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_WS_A = "workspace-a"
_WS_B = "workspace-b"
_NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
_DIGEST = "sha256:" + ("a" * 64)

_FORBIDDEN_CONTRACT_IMPORT_PREFIXES = (
    "intergrax.contracts.context_view",
    "intergrax.contracts.context_view_source_ports",
    "intergrax.contracts.context_view_composition",
    "intergrax.memory",
    "intergrax.rag",
    "intergrax.ucl",
    "intergrax.runtime.nexus",
)

_FORBIDDEN_DEFAULT_IMPORT_PREFIXES = _FORBIDDEN_CONTRACT_IMPORT_PREFIXES + (
    "intergrax.collaborative_work.in_memory_repository",
    "intergrax.collaborative_work.sqlite_repository",
    "intergrax.collaborative_work.postgresql_repository",
)


def _identity(
    *,
    tenant_id: str = _TENANT_A,
    user_id: str = "principal-1",
) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def _scope(
    *,
    tenant_id: str = _TENANT_A,
    workspace_id: str = _WS_A,
    work_item_id: str | None = None,
    work_artifact_id: str | None = None,
    work_artifact_version_id: str | None = None,
) -> CollaborativeWorkReferenceReadScope:
    return CollaborativeWorkReferenceReadScope(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        work_item_id=work_item_id,
        work_artifact_id=work_artifact_id,
        work_artifact_version_id=work_artifact_version_id,
    )


def _query(
    *,
    entity_kinds: frozenset[CollaborativeWorkReferenceEntityKind] | None = None,
    limit: int = 50,
    version_selection: CollaborativeWorkVersionSelection = (
        CollaborativeWorkVersionSelection.CURRENT_ONLY
    ),
) -> CollaborativeWorkReferenceReadQuery:
    kinds = entity_kinds or frozenset({CollaborativeWorkReferenceEntityKind.WORK_ITEM})
    return CollaborativeWorkReferenceReadQuery(
        entity_kinds=kinds,
        limit=limit,
        version_selection=version_selection,
    )


def _content_ref() -> ArtifactContentRef:
    return ArtifactContentRef.model_validate(
        {
            "content_ref": "content://tenant-a/workspace-a/body-1",
            "media_type": "application/json",
            "integrity_digest": _DIGEST,
        }
    )


def _seed_work_item(
    repo: InMemoryWorkItemRepository,
    *,
    work_item_id: str,
    tenant_id: str = _TENANT_A,
    workspace_id: str = _WS_A,
) -> None:
    repo.create(
        CreateWorkItemCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_item_id=work_item_id,
            created_by_principal_id="principal-1",
            created_at=_NOW,
            updated_at=_NOW,
        )
    )


def _seed_artifact(
    *,
    work_item_id: str,
    work_artifact_id: str,
    version_id: str,
    tenant_id: str = _TENANT_A,
    workspace_id: str = _WS_A,
) -> InMemoryArtifactRepositories:
    bundle = open_in_memory_artifact_repositories()
    bundle.publication.create_artifact_with_initial_version(
        CreateArtifactWithInitialVersionCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_item_id=work_item_id,
            work_artifact_id=work_artifact_id,
            work_artifact_version_id=version_id,
            created_by_principal_id="principal-1",
            published_by_principal_id="principal-1",
            content_ref=_content_ref(),
            artifact_created_at=_NOW,
            artifact_updated_at=_NOW,
            version_created_at=_NOW,
            version_published_at=_NOW,
            execution=None,
        )
    )
    return bundle


def _reader(
    *,
    work_item_repo: InMemoryWorkItemRepository | None = None,
    artifact_bundle: InMemoryArtifactRepositories | None = None,
    tenant_id: str = _TENANT_A,
    workspace_id: str = _WS_A,
) -> DefaultCollaborativeWorkReferenceReader:
    work_items = work_item_repo or InMemoryWorkItemRepository()
    bundle = artifact_bundle or open_in_memory_artifact_repositories()
    catalog = RepositoryBackedCollaborativeWorkReferenceCatalog(
        work_item_repository=work_items,
        work_artifact_repository=bundle.artifact,
        work_artifact_version_repository=bundle.version,
    )
    return DefaultCollaborativeWorkReferenceReader(
        catalog=catalog,
        capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ),
    )


def test_contract_request_immutability_and_query_bounds() -> None:
    _ = CollaborativeWorkReferenceReadRequest(scope=_scope(), query=_query())
    with pytest.raises(CollaborativeWorkReferenceReadScopeError):
        CollaborativeWorkReferenceReadQuery(
            entity_kinds=frozenset({CollaborativeWorkReferenceEntityKind.WORK_ITEM}),
            limit=0,
        )
    with pytest.raises(CollaborativeWorkReferenceReadScopeError):
        CollaborativeWorkReferenceReadQuery(
            entity_kinds=frozenset({CollaborativeWorkReferenceEntityKind.WORK_ITEM}),
            limit=COLLABORATIVE_WORK_REFERENCE_READ_MAX_LIMIT + 1,
        )
    with pytest.raises(CollaborativeWorkReferenceReadScopeError):
        _scope(tenant_id="")


def test_validate_identity_tenant_mismatch_scope_rejected() -> None:
    identity = _identity(tenant_id=_TENANT_A)
    request = CollaborativeWorkReferenceReadRequest(
        scope=_scope(tenant_id=_TENANT_B),
        query=_query(),
    )
    assert (
        validate_collaborative_work_reference_read_request(identity, request)
        is CollaborativeWorkReferenceReadOutcome.SCOPE_REJECTED
    )


def test_result_reference_only_no_payload_fields() -> None:
    ref = CollaborativeWorkItemCanonicalRef(
        tenant_id=_TENANT_A,
        workspace_id=_WS_A,
        work_item_id="wi-1",
        state=WorkItemState.OPEN,
    )
    payload = dataclasses.asdict(ref)
    forbidden = {
        "content",
        "title",
        "description",
        "body",
        "payload",
        "content_ref",
        "integrity_digest",
    }
    assert forbidden.isdisjoint(payload.keys())


def test_valid_work_item_returned() -> None:
    work_items = InMemoryWorkItemRepository()
    _seed_work_item(work_items, work_item_id="wi-1")
    reader = _reader(work_item_repo=work_items)
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(work_item_id="wi-1"),
            query=_query(),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.OK
    assert len(result.references) == 1
    ref = result.references[0]
    assert isinstance(ref, CollaborativeWorkItemCanonicalRef)
    assert ref.work_item_id == "wi-1"


def test_wrong_tenant_scope_rejected() -> None:
    work_items = InMemoryWorkItemRepository()
    _seed_work_item(work_items, work_item_id="wi-1")
    reader = _reader(work_item_repo=work_items)
    result = reader.read_references(
        _identity(tenant_id=_TENANT_A),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(tenant_id=_TENANT_B),
            query=_query(),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.SCOPE_REJECTED


def test_wrong_workspace_empty_ok() -> None:
    work_items = InMemoryWorkItemRepository()
    _seed_work_item(work_items, work_item_id="wi-1", workspace_id=_WS_B)
    reader = _reader(work_item_repo=work_items)
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(work_item_id="wi-1"),
            query=_query(),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.OK
    assert result.references == ()


def test_forged_work_item_excludes_artifact() -> None:
    work_items = InMemoryWorkItemRepository()
    _seed_work_item(work_items, work_item_id="wi-a")
    _seed_work_item(work_items, work_item_id="wi-b")
    bundle = _seed_artifact(
        work_item_id="wi-b",
        work_artifact_id="art-y",
        version_id="ver-1",
    )
    reader = _reader(work_item_repo=work_items, artifact_bundle=bundle)
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(work_item_id="wi-a", work_artifact_id="art-y"),
            query=_query(
                entity_kinds=frozenset(
                    {CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT}
                )
            ),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.OK
    assert result.references == ()


def test_valid_artifact_and_version_returned() -> None:
    work_items = InMemoryWorkItemRepository()
    _seed_work_item(work_items, work_item_id="wi-1")
    bundle = _seed_artifact(
        work_item_id="wi-1",
        work_artifact_id="art-1",
        version_id="ver-1",
    )
    reader = _reader(work_item_repo=work_items, artifact_bundle=bundle)
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(
                work_item_id="wi-1",
                work_artifact_id="art-1",
                work_artifact_version_id="ver-1",
            ),
            query=_query(
                entity_kinds=frozenset(
                    {
                        CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT,
                        CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT_VERSION,
                    }
                )
            ),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.OK
    kinds = {type(ref) for ref in result.references}
    assert CollaborativeWorkArtifactCanonicalRef in kinds
    assert WorkArtifactVersionRef in kinds


def test_forged_artifact_parent_excludes_version() -> None:
    work_items = InMemoryWorkItemRepository()
    _seed_work_item(work_items, work_item_id="wi-1")
    bundle = _seed_artifact(
        work_item_id="wi-1",
        work_artifact_id="art-y",
        version_id="ver-z",
    )
    reader = _reader(work_item_repo=work_items, artifact_bundle=bundle)
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(
                work_item_id="wi-1",
                work_artifact_id="art-x",
                work_artifact_version_id="ver-z",
            ),
            query=_query(
                entity_kinds=frozenset(
                    {CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT_VERSION}
                )
            ),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.OK
    assert result.references == ()


def test_current_only_hides_historical_versions() -> None:
    work_items = InMemoryWorkItemRepository()
    _seed_work_item(work_items, work_item_id="wi-1")
    bundle = open_in_memory_artifact_repositories()
    bundle.publication.create_artifact_with_initial_version(
        CreateArtifactWithInitialVersionCommand(
            tenant_id=_TENANT_A,
            workspace_id=_WS_A,
            work_item_id="wi-1",
            work_artifact_id="art-1",
            work_artifact_version_id="ver-1",
            created_by_principal_id="principal-1",
            published_by_principal_id="principal-1",
            content_ref=_content_ref(),
            artifact_created_at=_NOW,
            artifact_updated_at=_NOW,
            version_created_at=_NOW,
            version_published_at=_NOW,
            execution=None,
        )
    )
    bundle.publication.publish_version(
        PublishWorkArtifactVersionCommand(
            tenant_id=_TENANT_A,
            workspace_id=_WS_A,
            work_item_id="wi-1",
            work_artifact_id="art-1",
            work_artifact_version_id="ver-2",
            expected_revision=0,
            created_by_principal_id="principal-1",
            published_by_principal_id="principal-1",
            content_ref=_content_ref(),
            created_at=_NOW + timedelta(minutes=1),
            published_at=_NOW + timedelta(minutes=2),
            artifact_updated_at=_NOW + timedelta(minutes=2),
        )
    )
    reader = _reader(work_item_repo=work_items, artifact_bundle=bundle)
    current = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(work_item_id="wi-1", work_artifact_id="art-1"),
            query=_query(
                entity_kinds=frozenset(
                    {CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT_VERSION}
                ),
                version_selection=CollaborativeWorkVersionSelection.CURRENT_ONLY,
            ),
        ),
    )
    assert current.outcome is CollaborativeWorkReferenceReadOutcome.OK
    assert len(current.references) == 1
    version_ref = current.references[0]
    assert isinstance(version_ref, WorkArtifactVersionRef)
    assert version_ref.work_artifact_version_id == "ver-2"

    historical = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(work_item_id="wi-1", work_artifact_id="art-1"),
            query=_query(
                entity_kinds=frozenset(
                    {CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT_VERSION}
                ),
                version_selection=CollaborativeWorkVersionSelection.INCLUDE_HISTORICAL,
            ),
        ),
    )
    assert historical.outcome is CollaborativeWorkReferenceReadOutcome.OK
    assert len(historical.references) == 2


def test_deterministic_ordering_and_limit_after_full_scope() -> None:
    work_items = InMemoryWorkItemRepository()
    for ident in ("wi-c", "wi-a", "wi-b"):
        _seed_work_item(work_items, work_item_id=ident)
    reader = _reader(work_item_repo=work_items)
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(),
            query=_query(limit=2),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.OK
    ids = [ref.work_item_id for ref in result.references]
    assert ids == ["wi-a", "wi-b"]


class _OutOfScopeCatalog:
    def list_scoped_references(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> tuple[CollaborativeWorkScopedReferenceListing, ...]:
        return (
            CollaborativeWorkScopedReferenceListing(
                entity_kind="work_item",
                tenant_id="other-tenant",
                workspace_id=query.workspace_id,
                work_item_id="wi-forged",
                work_item_state=WorkItemState.OPEN,
            ),
        )


class _WrongEntityKindCatalog:
    def list_scoped_references(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> tuple[CollaborativeWorkScopedReferenceListing, ...]:
        return (
            CollaborativeWorkScopedReferenceListing(
                entity_kind="work_item",
                tenant_id=query.tenant_id,
                workspace_id=query.workspace_id,
                work_item_id="wi-1",
                work_item_state=WorkItemState.OPEN,
            ),
        )


def test_provider_wrong_entity_kind_fail_closed() -> None:
    reader = DefaultCollaborativeWorkReferenceReader(
        catalog=_WrongEntityKindCatalog(),
        capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
            tenant_id=_TENANT_A,
            workspace_id=_WS_A,
        ),
    )
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(),
            query=_query(
                entity_kinds=frozenset(
                    {CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT_VERSION}
                ),
            ),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.UNAVAILABLE
    assert result.reason == "catalog_contract_violation"


class _LimitOverflowCatalog:
    def list_scoped_references(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> tuple[CollaborativeWorkScopedReferenceListing, ...]:
        row = CollaborativeWorkScopedReferenceListing(
            entity_kind="work_item",
            tenant_id=query.tenant_id,
            workspace_id=query.workspace_id,
            work_item_id="wi-a",
            work_item_state=WorkItemState.OPEN,
        )
        row_b = CollaborativeWorkScopedReferenceListing(
            entity_kind="work_item",
            tenant_id=query.tenant_id,
            workspace_id=query.workspace_id,
            work_item_id="wi-b",
            work_item_state=WorkItemState.OPEN,
        )
        return (row, row_b)


def test_provider_limit_overflow_fail_closed() -> None:
    reader = DefaultCollaborativeWorkReferenceReader(
        catalog=_LimitOverflowCatalog(),
        capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
            tenant_id=_TENANT_A,
            workspace_id=_WS_A,
        ),
    )
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(),
            query=_query(limit=1),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.UNAVAILABLE
    assert result.reason == "catalog_contract_violation"


class _WrongArtifactFilterCatalog:
    def list_scoped_references(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> tuple[CollaborativeWorkScopedReferenceListing, ...]:
        return (
            CollaborativeWorkScopedReferenceListing(
                entity_kind="work_artifact_version",
                tenant_id=query.tenant_id,
                workspace_id=query.workspace_id,
                work_item_id="wi-1",
                work_artifact_id="art-b",
                work_artifact_version_id="ver-1",
            ),
        )


def test_provider_wrong_artifact_filter_fail_closed() -> None:
    reader = DefaultCollaborativeWorkReferenceReader(
        catalog=_WrongArtifactFilterCatalog(),
        capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
            tenant_id=_TENANT_A,
            workspace_id=_WS_A,
        ),
    )
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(work_item_id="wi-1", work_artifact_id="art-a"),
            query=_query(
                entity_kinds=frozenset(
                    {CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT_VERSION}
                ),
            ),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.UNAVAILABLE
    assert result.reason == "catalog_contract_violation"


class _WrongVersionFilterCatalog:
    def list_scoped_references(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> tuple[CollaborativeWorkScopedReferenceListing, ...]:
        return (
            CollaborativeWorkScopedReferenceListing(
                entity_kind="work_artifact_version",
                tenant_id=query.tenant_id,
                workspace_id=query.workspace_id,
                work_item_id="wi-1",
                work_artifact_id="art-1",
                work_artifact_version_id="ver-2",
            ),
        )


def test_provider_wrong_version_filter_fail_closed() -> None:
    reader = DefaultCollaborativeWorkReferenceReader(
        catalog=_WrongVersionFilterCatalog(),
        capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
            tenant_id=_TENANT_A,
            workspace_id=_WS_A,
        ),
    )
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(
                work_item_id="wi-1",
                work_artifact_id="art-1",
                work_artifact_version_id="ver-1",
            ),
            query=_query(
                entity_kinds=frozenset(
                    {CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT_VERSION}
                ),
            ),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.UNAVAILABLE
    assert result.reason == "catalog_contract_violation"


class _InconsistentParentCatalog:
    def list_scoped_references(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> tuple[CollaborativeWorkScopedReferenceListing, ...]:
        return (
            CollaborativeWorkScopedReferenceListing(
                entity_kind="work_artifact_version",
                tenant_id=query.tenant_id,
                workspace_id=query.workspace_id,
                work_item_id="wi-other",
                work_artifact_id="art-1",
                work_artifact_version_id="ver-1",
            ),
        )


def test_provider_inconsistent_parent_chain_fail_closed() -> None:
    reader = DefaultCollaborativeWorkReferenceReader(
        catalog=_InconsistentParentCatalog(),
        capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
            tenant_id=_TENANT_A,
            workspace_id=_WS_A,
        ),
    )
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(work_item_id="wi-1", work_artifact_id="art-1"),
            query=_query(
                entity_kinds=frozenset(
                    {CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT_VERSION}
                ),
            ),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.UNAVAILABLE
    assert result.reason == "catalog_contract_violation"


class _IgnoresHistoricalCatalog:
    def list_scoped_references(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> tuple[CollaborativeWorkScopedReferenceListing, ...]:
        if query.include_historical:
            return ()
        return (
            CollaborativeWorkScopedReferenceListing(
                entity_kind="work_artifact_version",
                tenant_id=query.tenant_id,
                workspace_id=query.workspace_id,
                work_item_id="wi-1",
                work_artifact_id="art-1",
                work_artifact_version_id="ver-historical",
            ),
        )


def test_custom_catalog_version_selection_is_catalog_contract() -> None:
    """Reader cannot prove CURRENT_ONLY without aggregate pointer evidence."""
    reader = DefaultCollaborativeWorkReferenceReader(
        catalog=_IgnoresHistoricalCatalog(),
        capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
            tenant_id=_TENANT_A,
            workspace_id=_WS_A,
        ),
    )
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(
            scope=_scope(work_item_id="wi-1", work_artifact_id="art-1"),
            query=_query(
                entity_kinds=frozenset(
                    {CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT_VERSION}
                ),
                version_selection=CollaborativeWorkVersionSelection.CURRENT_ONLY,
            ),
        ),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.OK
    assert len(result.references) == 1


def test_plugin_out_of_scope_listing_fail_closed() -> None:
    reader = DefaultCollaborativeWorkReferenceReader(
        catalog=_OutOfScopeCatalog(),
        capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
            tenant_id=_TENANT_A,
            workspace_id=_WS_A,
        ),
    )
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(scope=_scope(), query=_query()),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.UNAVAILABLE
    assert result.reason == "catalog_contract_violation"


def test_listing_work_item_missing_work_item_state_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="WorkItemState"):
        CollaborativeWorkScopedReferenceListing(
            entity_kind="work_item",
            tenant_id=_TENANT_A,
            workspace_id=_WS_A,
            work_item_id="wi-1",
            work_item_state=None,
        )


def test_listing_work_artifact_missing_current_version_id_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="current_version"):
        CollaborativeWorkScopedReferenceListing(
            entity_kind="work_artifact",
            tenant_id=_TENANT_A,
            workspace_id=_WS_A,
            work_item_id="wi-1",
            work_artifact_id="art-1",
            current_version_id=None,
        )


class _CustomCollaborativeWorkReferenceReader:
    def read_references(
        self,
        identity: RequestIdentity,
        request: CollaborativeWorkReferenceReadRequest,
    ) -> CollaborativeWorkReferenceReadResult:
        if request.scope.tenant_id != identity.tenant_id:
            return CollaborativeWorkReferenceReadResult(
                outcome=CollaborativeWorkReferenceReadOutcome.SCOPE_REJECTED,
            )
        ref = CollaborativeWorkItemCanonicalRef(
            tenant_id=request.scope.tenant_id,
            workspace_id=request.scope.workspace_id,
            work_item_id="custom-wi",
            state=WorkItemState.ACTIVE,
        )
        return CollaborativeWorkReferenceReadResult(
            outcome=CollaborativeWorkReferenceReadOutcome.OK,
            references=(ref,),
        )


def test_pluginability_custom_port_without_default_runtime() -> None:
    port: CollaborativeWorkReferenceReadPort = _CustomCollaborativeWorkReferenceReader()
    assert isinstance(port, CollaborativeWorkReferenceReadPort)


class _FakeExternalCatalog:
    def list_scoped_references(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> tuple[CollaborativeWorkScopedReferenceListing, ...]:
        return (
            CollaborativeWorkScopedReferenceListing(
                entity_kind="work_item",
                tenant_id=query.tenant_id,
                workspace_id=query.workspace_id,
                work_item_id="external-wi",
                work_item_state=WorkItemState.OPEN,
            ),
        )


def test_external_catalog_without_concrete_repository() -> None:
    catalog: CollaborativeWorkScopedReferenceCatalog = _FakeExternalCatalog()
    reader = DefaultCollaborativeWorkReferenceReader(
        catalog=catalog,
        capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
            tenant_id=_TENANT_A,
            workspace_id=_WS_A,
        ),
    )
    result = reader.read_references(
        _identity(),
        CollaborativeWorkReferenceReadRequest(scope=_scope(), query=_query()),
    )
    assert result.outcome is CollaborativeWorkReferenceReadOutcome.OK
    assert result.references[0].work_item_id == "external-wi"


def _forbidden_imports(module_path: Path, prefixes: tuple[str, ...]) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in prefixes:
                    if alias.name == prefix or alias.name.startswith(prefix + "."):
                        violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for prefix in prefixes:
                if node.module == prefix or node.module.startswith(prefix + "."):
                    violations.append(node.module)
    return violations


def test_architecture_gate_public_contract_imports() -> None:
    violations = _forbidden_imports(_CONTRACT, _FORBIDDEN_CONTRACT_IMPORT_PREFIXES)
    assert violations == []


def test_architecture_gate_default_reader_imports() -> None:
    violations = _forbidden_imports(_DEFAULT_READER, _FORBIDDEN_DEFAULT_IMPORT_PREFIXES)
    assert violations == []


def test_architecture_gate_catalog_imports_no_concrete_repositories() -> None:
    violations = _forbidden_imports(
        _CATALOG,
        (
            "intergrax.collaborative_work.in_memory_repository",
            "intergrax.collaborative_work.sqlite_repository",
            "intergrax.collaborative_work.postgresql_repository",
        ),
    )
    assert violations == []
