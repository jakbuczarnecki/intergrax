# © Artur Czarnecki. All rights reserved.

"""MP-5F-B3B-C1 — resource-scoped catalog query before limit."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pytest

from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.runtime.context_lifecycle import (
    ArtifactCompressionTarget,
    ArtifactLookupKey,
    ArtifactSourceRange,
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    InMemoryOptimizationArtifactRepository,
    OptimizationArtifactRepository,
    OptimizationArtifactType,
    ReusableArtifactStatus,
    ReusableOptimizationArtifact,
    SQLiteOptimizationArtifactRepository,
    StoredOptimizationArtifact,
    build_optimization_artifact_reference,
    UclArtifactOwnership,
    UclArtifactOwnershipKind,
    UclArtifactOwnershipScope,
    compute_artifact_content_hash,
)
from intergrax.runtime.context_lifecycle.default_ucl_reference_reader import (
    DefaultUclReferenceReader,
    UclReferenceReadCapabilityBinding,
)
from intergrax.runtime.context_lifecycle.repository import (
    OptimizationArtifactReference,
    OptimizationArtifactScopedReferenceCatalog,
    OptimizationArtifactScopedReferenceQuery,
    ScopedOptimizationArtifactListing,
)
from intergrax.ucl.contracts.ucl_reference_read import (
    UclReferenceLifecycleSelection,
    UclReferenceReadOutcome,
    UclReferenceReadQuery,
    UclReferenceReadRequest,
    UclReferenceReadScope,
    UclScopedResourceRef,
)

pytestmark = pytest.mark.gate


class _RepositoryWithScopedCatalog(
    OptimizationArtifactRepository,
    OptimizationArtifactScopedReferenceCatalog,
    Protocol,
):
    """Test-local intersection: mutable repository + scoped catalog enumeration."""


_BASE_TIME = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)


def _identity(tenant_id: str = "tenant-a") -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="user-1",
        principal_type=PrincipalType.USER,
        auth_subject="user-1",
    )


def _lookup_key(
    *,
    tenant_id: str = "tenant-a",
    context_scope_id: str = "ctx-x",
    source_content_hash: str = "hash-abc",
    source_refs: tuple[str, ...] = ("msg-1", "msg-2"),
    source_range: ArtifactSourceRange | None = None,
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
        source_range=source_range,
    )


def _stored(
    *,
    artifact_id: str,
    workspace_id: str,
    context_scope_id: str = "ctx-x",
    tenant_id: str = "tenant-a",
    source_refs: tuple[str, ...] = ("msg-1", "msg-2"),
    source_content_hash: str | None = None,
    ownership_kind: UclArtifactOwnershipKind = UclArtifactOwnershipKind.WORKSPACE,
) -> StoredOptimizationArtifact:
    key = _lookup_key(
        tenant_id=tenant_id,
        context_scope_id=context_scope_id,
        source_refs=source_refs,
        source_content_hash=source_content_hash if source_content_hash is not None else "hash-abc",
    )
    if ownership_kind is UclArtifactOwnershipKind.WORKSPACE:
        ownership = UclArtifactOwnership.for_workspace(
            UclArtifactOwnershipScope(tenant_id=tenant_id, workspace_id=workspace_id),
        )
    else:
        ownership = UclArtifactOwnership.legacy_unknown()
    payload = f"payload-{artifact_id}".encode()
    metadata = ReusableOptimizationArtifact(
        artifact_id=artifact_id,
        lookup_key=key,
        ownership=ownership,
        artifact_content_hash=compute_artifact_content_hash(payload),
        created_at=_BASE_TIME,
        created_by_executor="executor.message_sequence",
        validation=ArtifactValidationSummary(
            status=ArtifactValidationStatus.PASSED,
            validation_contract_version="validation-v1",
            validated_at=_BASE_TIME,
        ),
        status=ReusableArtifactStatus.VALIDATED,
    )
    return StoredOptimizationArtifact(
        metadata=metadata,
        payload=payload,
        media_type="application/octet-stream",
    )


def _publish(repository: OptimizationArtifactRepository, artifact: StoredOptimizationArtifact) -> None:
    key = artifact.metadata.lookup_key
    scope = artifact.metadata.ownership.scope
    if scope is None:
        raise AssertionError("workspace publish requires WORKSPACE ownership")
    reservation = repository.try_acquire_creation_reservation(
        key,
        ownership=scope,
        owner_operation_id="op-b3b-c1",
        lease_seconds=60,
    )
    assert reservation.reservation is not None
    repository.store_validated_artifact(
        reservation=reservation.reservation,
        artifact=artifact,
    )


def _reader(
    catalog: OptimizationArtifactScopedReferenceCatalog,
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "ws-a",
    context_scope_id: str = "ctx-x",
) -> DefaultUclReferenceReader:
    return DefaultUclReferenceReader(
        catalog=catalog,
        capability_binding=UclReferenceReadCapabilityBinding(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            context_scope_id=context_scope_id,
        ),
    )


@pytest.fixture(params=("memory", "sqlite"))
def catalog_repository(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[_RepositoryWithScopedCatalog]:
    if request.param == "memory":
        repo: _RepositoryWithScopedCatalog = InMemoryOptimizationArtifactRepository()
    else:
        repo = SQLiteOptimizationArtifactRepository(str(tmp_path / "b3b-c1-read.sqlite"))
    yield repo
    repo.close()


@pytest.mark.asyncio
async def test_resource_scope_before_limit_blocker_regression(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    for index in range(50):
        _publish(
            catalog_repository,
            _stored(
                artifact_id=f"b-{index}",
                workspace_id="ws-b",
                source_refs=("other",),
                source_content_hash=f"hash-b-{index}",
            ),
        )
    for index in range(2):
        _publish(
            catalog_repository,
            _stored(
                artifact_id=f"a-wanted-{index}",
                workspace_id="ws-a",
                source_refs=("wanted",),
                source_content_hash=f"hash-wanted-{index}",
            ),
        )
    reader = _reader(catalog_repository)
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=UclReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-a",
                context_scope_id="ctx-x",
                resource=UclScopedResourceRef(resource_kind="source_ref", resource_id="wanted"),
            ),
            query=UclReferenceReadQuery(limit=2),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert len(result.references) == 2
    assert {ref.artifact_id for ref in result.references} == {"a-wanted-0", "a-wanted-1"}


@pytest.mark.asyncio
async def test_matching_rows_after_nonmatching_still_returned_up_to_limit(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    _publish(
        catalog_repository,
        _stored(artifact_id="artifact-1", workspace_id="ws-a", source_refs=("other",)),
    )
    _publish(
        catalog_repository,
        _stored(
            artifact_id="artifact-2",
            workspace_id="ws-a",
            source_refs=("other",),
            source_content_hash="hash-2",
        ),
    )
    _publish(
        catalog_repository,
        _stored(
            artifact_id="artifact-3",
            workspace_id="ws-a",
            source_refs=("wanted",),
            source_content_hash="hash-3",
        ),
    )
    _publish(
        catalog_repository,
        _stored(
            artifact_id="artifact-4",
            workspace_id="ws-a",
            source_refs=("wanted",),
            source_content_hash="hash-4",
        ),
    )
    reader = _reader(catalog_repository)
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=UclReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-a",
                context_scope_id="ctx-x",
                resource=UclScopedResourceRef(resource_kind="source_ref", resource_id="wanted"),
            ),
            query=UclReferenceReadQuery(limit=2),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert [ref.artifact_id for ref in result.references] == ["artifact-3", "artifact-4"]


@pytest.mark.asyncio
async def test_no_resource_preserves_workspace_context_behavior(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    _publish(catalog_repository, _stored(artifact_id="artifact-a", workspace_id="ws-a"))
    reader = _reader(catalog_repository)
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=UclReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-a",
                context_scope_id="ctx-x",
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert len(result.references) == 1


@pytest.mark.asyncio
async def test_cross_scope_source_ref_excluded(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    _publish(
        catalog_repository,
        _stored(
            artifact_id="wrong-ws",
            workspace_id="ws-b",
            source_refs=("wanted",),
            source_content_hash="hash-ws",
        ),
    )
    _publish(
        catalog_repository,
        _stored(
            artifact_id="wrong-ctx",
            workspace_id="ws-a",
            context_scope_id="ctx-y",
            source_refs=("wanted",),
            source_content_hash="hash-ctx",
        ),
    )
    reader = _reader(catalog_repository)
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=UclReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-a",
                context_scope_id="ctx-x",
                resource=UclScopedResourceRef(resource_kind="source_ref", resource_id="wanted"),
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert result.references == ()


@pytest.mark.asyncio
async def test_historical_read_respects_source_ref(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    active = _stored(
        artifact_id="active-wanted",
        workspace_id="ws-a",
        source_refs=("wanted",),
    )
    _publish(catalog_repository, active)
    retired = _stored(
        artifact_id="retired-wanted",
        workspace_id="ws-a",
        source_refs=("wanted",),
        source_content_hash="hash-retired",
    )
    _publish(catalog_repository, retired)
    catalog_repository.invalidate_artifact(
        build_optimization_artifact_reference(retired),
        reason="superseded",
    )
    reader = _reader(catalog_repository)
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=UclReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-a",
                context_scope_id="ctx-x",
                resource=UclScopedResourceRef(resource_kind="source_ref", resource_id="wanted"),
            ),
            query=UclReferenceReadQuery(
                lifecycle_selection=UclReferenceLifecycleSelection.INCLUDE_HISTORICAL,
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert {ref.artifact_id for ref in result.references} == {"active-wanted", "retired-wanted"}


class _RecordingCatalog:
    last_query: OptimizationArtifactScopedReferenceQuery | None = None

    def list_scoped_artifact_references(
        self,
        query: OptimizationArtifactScopedReferenceQuery,
    ) -> tuple[ScopedOptimizationArtifactListing, ...]:
        _RecordingCatalog.last_query = query
        return ()


@pytest.mark.asyncio
async def test_custom_catalog_receives_full_scoped_query() -> None:
    catalog: OptimizationArtifactScopedReferenceCatalog = _RecordingCatalog()
    reader = DefaultUclReferenceReader(
        catalog=catalog,
        capability_binding=UclReferenceReadCapabilityBinding(
            tenant_id="tenant-a",
            workspace_id="ws-custom",
            context_scope_id="ctx-x",
        ),
    )
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=UclReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-custom",
                context_scope_id="ctx-x",
                resource=UclScopedResourceRef(resource_kind="source_ref", resource_id="src-1"),
            ),
            query=UclReferenceReadQuery(limit=3, lifecycle_selection=UclReferenceLifecycleSelection.INCLUDE_HISTORICAL),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    query = _RecordingCatalog.last_query
    assert query is not None
    assert query.tenant_id == "tenant-a"
    assert query.workspace_id == "ws-custom"
    assert query.context_scope_id == "ctx-x"
    assert query.source_ref == "src-1"
    assert query.limit == 3
    assert query.include_historical is True


class _WrongWorkspaceCatalog:
    def list_scoped_artifact_references(
        self,
        query: OptimizationArtifactScopedReferenceQuery,
    ) -> tuple[ScopedOptimizationArtifactListing, ...]:
        return (
            ScopedOptimizationArtifactListing(
                reference=OptimizationArtifactReference(
                    tenant_id=query.tenant_id,
                    artifact_id="x",
                    artifact_lookup_key_hash="hash",
                    artifact_content_hash="content",
                    artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
                    context_scope_id=query.context_scope_id,
                    workspace_id="ws-other",
                ),
                context_scope_id=query.context_scope_id,
                lifecycle_status=ReusableArtifactStatus.VALIDATED,
                source_refs=("src-1",),
            ),
        )


@pytest.mark.asyncio
async def test_custom_catalog_out_of_scope_listing_fails_closed() -> None:
    reader = DefaultUclReferenceReader(
        catalog=_WrongWorkspaceCatalog(),
        capability_binding=UclReferenceReadCapabilityBinding(
            tenant_id="tenant-a",
            workspace_id="ws-a",
            context_scope_id="ctx-x",
        ),
    )
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=UclReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-a",
                context_scope_id="ctx-x",
                resource=UclScopedResourceRef(resource_kind="source_ref", resource_id="src-1"),
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.UNAVAILABLE
    assert result.reason == "catalog_contract_violation"


def test_in_memory_sqlite_parity_same_order(tmp_path: Path) -> None:
    memory = InMemoryOptimizationArtifactRepository()
    sqlite = SQLiteOptimizationArtifactRepository(str(tmp_path / "parity.sqlite"))
    for repo in (memory, sqlite):
        for artifact_id, source_ref in (
            ("artifact-1", "other"),
            ("artifact-2", "other"),
            ("artifact-3", "wanted"),
            ("artifact-4", "wanted"),
        ):
            _publish(
                repo,
                _stored(
                    artifact_id=artifact_id,
                    workspace_id="ws-a",
                    source_refs=(source_ref,),
                    source_content_hash=f"hash-{artifact_id}",
                ),
            )
    query = OptimizationArtifactScopedReferenceQuery(
        tenant_id="tenant-a",
        workspace_id="ws-a",
        context_scope_id="ctx-x",
        source_ref="wanted",
        limit=2,
    )
    memory_ids = tuple(
        row.reference.artifact_id for row in memory.list_scoped_artifact_references(query)
    )
    sqlite_ids = tuple(
        row.reference.artifact_id for row in sqlite.list_scoped_artifact_references(query)
    )
    assert memory_ids == sqlite_ids == ("artifact-3", "artifact-4")
    memory.close()
    sqlite.close()


def test_source_range_artifact_excluded_for_source_ref_query(tmp_path: Path) -> None:
    repo = InMemoryOptimizationArtifactRepository()
    key = _lookup_key(
        source_refs=(),
        source_range=ArtifactSourceRange(start_sequence=0, end_sequence=3),
        source_content_hash="range-hash",
    )
    ownership = UclArtifactOwnership.for_workspace(
        UclArtifactOwnershipScope(tenant_id="tenant-a", workspace_id="ws-a"),
    )
    payload = b"range-payload"
    metadata = ReusableOptimizationArtifact(
        artifact_id="range-only",
        lookup_key=key,
        ownership=ownership,
        artifact_content_hash=compute_artifact_content_hash(payload),
        created_at=_BASE_TIME,
        created_by_executor="executor.message_sequence",
        validation=ArtifactValidationSummary(
            status=ArtifactValidationStatus.PASSED,
            validation_contract_version="validation-v1",
            validated_at=_BASE_TIME,
        ),
        status=ReusableArtifactStatus.VALIDATED,
    )
    _publish(repo, StoredOptimizationArtifact(metadata=metadata, payload=payload, media_type="application/octet-stream"))
    query = OptimizationArtifactScopedReferenceQuery(
        tenant_id="tenant-a",
        workspace_id="ws-a",
        context_scope_id="ctx-x",
        source_ref="wanted",
        limit=10,
    )
    assert repo.list_scoped_artifact_references(query) == ()
    repo.close()
