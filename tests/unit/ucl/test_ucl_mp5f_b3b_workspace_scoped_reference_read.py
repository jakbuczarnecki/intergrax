# © Artur Czarnecki. All rights reserved.

"""MP-5F-B3B — workspace-scoped UCL reference read certification."""

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
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    InMemoryOptimizationArtifactRepository,
    OptimizationArtifactRepository,
    OptimizationArtifactType,
    ReusableArtifactStatus,
    ReusableOptimizationArtifact,
    SQLiteOptimizationArtifactRepository,
    StoredOptimizationArtifact,
    UclArtifactOwnership,
    UclArtifactOwnershipKind,
    UclArtifactOwnershipScope,
    build_optimization_artifact_reference,
    compute_artifact_content_hash,
)
from intergrax.runtime.context_lifecycle.default_ucl_reference_reader import (
    DefaultUclReferenceReader,
    UclReferenceReadCapabilityBinding,
    UclReferenceReadConfigurationError,
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
    UclReferenceReadPort,
    UclReferenceReadQuery,
    UclReferenceReadRequest,
    UclReferenceReadResult,
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
_DEFAULT_READER = (
    Path(__file__).resolve().parents[3]
    / "intergrax"
    / "runtime"
    / "context_lifecycle"
    / "default_ucl_reference_reader.py"
)


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
        owner_operation_id="op-b3b",
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
        repo = SQLiteOptimizationArtifactRepository(str(tmp_path / "b3b-read.sqlite"))
    yield repo
    repo.close()


@pytest.mark.asyncio
async def test_correct_tenant_workspace_context_returns_refs(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    artifact = _stored(artifact_id="artifact-a", workspace_id="ws-a")
    _publish(catalog_repository, artifact)
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
    ref = result.references[0]
    assert ref.tenant_id == "tenant-a"
    assert ref.workspace_id == "ws-a"
    assert ref.context_scope_id == "ctx-x"


@pytest.mark.asyncio
async def test_wrong_tenant_scope_rejected() -> None:
    reader = DefaultUclReferenceReader()
    result = await reader.read_references(
        _identity(tenant_id="tenant-a"),
        UclReferenceReadRequest(
            scope=UclReferenceReadScope(
                tenant_id="tenant-b",
                workspace_id="ws-a",
                context_scope_id="ctx-x",
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.SCOPE_REJECTED


@pytest.mark.asyncio
async def test_wrong_binding_workspace_rejected(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    _publish(catalog_repository, _stored(artifact_id="artifact-a", workspace_id="ws-a"))
    reader = _reader(catalog_repository, workspace_id="ws-a")
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=UclReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-b",
                context_scope_id="ctx-x",
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.SCOPE_REJECTED


@pytest.mark.asyncio
async def test_wrong_binding_context_rejected(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    _publish(catalog_repository, _stored(artifact_id="artifact-a", workspace_id="ws-a"))
    reader = _reader(catalog_repository, context_scope_id="ctx-x")
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=UclReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-a",
                context_scope_id="ctx-y",
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.SCOPE_REJECTED


@pytest.mark.asyncio
async def test_cross_workspace_same_context_isolated(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    _publish(catalog_repository, _stored(artifact_id="artifact-a", workspace_id="ws-a"))
    _publish(
        catalog_repository,
        _stored(artifact_id="artifact-b", workspace_id="ws-b"),
    )
    reader = _reader(catalog_repository, workspace_id="ws-a")
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
    assert {ref.artifact_id for ref in result.references} == {"artifact-a"}


@pytest.mark.asyncio
async def test_same_workspace_different_context_isolated(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    _publish(
        catalog_repository,
        _stored(artifact_id="artifact-x", workspace_id="ws-a", context_scope_id="ctx-x"),
    )
    _publish(
        catalog_repository,
        _stored(artifact_id="artifact-y", workspace_id="ws-a", context_scope_id="ctx-y"),
    )
    reader = _reader(catalog_repository, workspace_id="ws-a", context_scope_id="ctx-x")
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
    assert {ref.artifact_id for ref in result.references} == {"artifact-x"}


def test_legacy_unknown_cannot_enter_workspace_scoped_catalog(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    legacy = _stored(
        artifact_id="legacy-1",
        workspace_id="ws-a",
        ownership_kind=UclArtifactOwnershipKind.LEGACY_UNKNOWN,
    )
    key = legacy.metadata.lookup_key
    ownership = UclArtifactOwnershipScope(tenant_id=key.tenant_id, workspace_id="ws-a")
    reservation = catalog_repository.try_acquire_creation_reservation(
        key,
        ownership=ownership,
        owner_operation_id="op-legacy",
        lease_seconds=60,
    )
    assert reservation.reservation is not None
    with pytest.raises(ValueError, match="WORKSPACE ownership"):
        catalog_repository.store_validated_artifact(
            reservation=reservation.reservation,
            artifact=legacy,
        )


@pytest.mark.asyncio
async def test_historical_cross_workspace_excluded(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    artifact_a = _stored(artifact_id="artifact-a", workspace_id="ws-a")
    _publish(catalog_repository, artifact_a)
    reference = build_optimization_artifact_reference(artifact_a)
    catalog_repository.invalidate_artifact(reference, reason="superseded")
    _publish(catalog_repository, _stored(artifact_id="artifact-b", workspace_id="ws-b"))
    reader = _reader(catalog_repository, workspace_id="ws-a")
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=UclReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-a",
                context_scope_id="ctx-x",
            ),
            query=UclReferenceReadQuery(
                lifecycle_selection=UclReferenceLifecycleSelection.INCLUDE_HISTORICAL,
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert all(ref.workspace_id == "ws-a" for ref in result.references)
    assert "artifact-b" not in {ref.artifact_id for ref in result.references}


@pytest.mark.asyncio
async def test_resource_filter_with_workspace_scope(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    _publish(
        catalog_repository,
        _stored(
            artifact_id="artifact-a",
            workspace_id="ws-a",
            source_refs=("keep-me",),
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
                resource=UclScopedResourceRef(resource_kind="source_ref", resource_id="missing"),
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert result.references == ()


@pytest.mark.asyncio
async def test_valid_scope_empty_ok(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
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
    assert result.references == ()


def test_malformed_binding_missing_workspace_rejected() -> None:
    with pytest.raises(UclReferenceReadConfigurationError):
        UclReferenceReadCapabilityBinding(
            tenant_id="tenant-a",
            workspace_id="",
            context_scope_id="ctx-x",
        )


@pytest.mark.asyncio
async def test_limit_applied_after_workspace_scope(
    catalog_repository: _RepositoryWithScopedCatalog,
) -> None:
    for index in range(50):
        _publish(
            catalog_repository,
            _stored(
                artifact_id=f"b-{index}",
                workspace_id="ws-b",
                source_content_hash=f"hash-b-{index}",
            ),
        )
    for index in range(2):
        _publish(
            catalog_repository,
            _stored(
                artifact_id=f"a-{index}",
                workspace_id="ws-a",
                source_content_hash=f"hash-a-{index}",
            ),
        )
    reader = _reader(catalog_repository, workspace_id="ws-a")
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=UclReferenceReadScope(
                tenant_id="tenant-a",
                workspace_id="ws-a",
                context_scope_id="ctx-x",
            ),
            query=UclReferenceReadQuery(limit=2),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert len(result.references) == 2
    assert all(ref.workspace_id == "ws-a" for ref in result.references)


def test_default_backends_statically_implement_repository_and_catalog(
    tmp_path: Path,
) -> None:
    memory: _RepositoryWithScopedCatalog = InMemoryOptimizationArtifactRepository()
    memory.close()
    sqlite: _RepositoryWithScopedCatalog = SQLiteOptimizationArtifactRepository(
        str(tmp_path / "b3b-static-conformance.sqlite"),
    )
    sqlite.close()


class _CustomCatalog:
    def list_scoped_artifact_references(
        self,
        query: OptimizationArtifactScopedReferenceQuery,
    ) -> tuple[ScopedOptimizationArtifactListing, ...]:
        assert query.tenant_id == "tenant-a"
        assert query.workspace_id == "ws-custom"
        assert query.context_scope_id == "ctx-x"
        return ()


@pytest.mark.asyncio
async def test_custom_catalog_receives_workspace_query() -> None:
    catalog: OptimizationArtifactScopedReferenceCatalog = _CustomCatalog()
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
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK


class _CustomCatalogWithListing:
    def list_scoped_artifact_references(
        self,
        query: OptimizationArtifactScopedReferenceQuery,
    ) -> tuple[ScopedOptimizationArtifactListing, ...]:
        return (
            ScopedOptimizationArtifactListing(
                reference=OptimizationArtifactReference(
                    tenant_id=query.tenant_id,
                    artifact_id="custom-artifact-1",
                    artifact_lookup_key_hash="lookup-hash-1",
                    artifact_content_hash="content-hash-1",
                    artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
                    context_scope_id=query.context_scope_id,
                    workspace_id=query.workspace_id,
                ),
                context_scope_id=query.context_scope_id,
                lifecycle_status=ReusableArtifactStatus.VALIDATED,
                source_refs=("msg-1",),
            ),
        )


@pytest.mark.asyncio
async def test_custom_catalog_listing_converted_to_canonical_ref() -> None:
    catalog: OptimizationArtifactScopedReferenceCatalog = _CustomCatalogWithListing()
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
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert len(result.references) == 1
    ref = result.references[0]
    assert ref.artifact_id == "custom-artifact-1"
    assert ref.tenant_id == "tenant-a"
    assert ref.workspace_id == "ws-custom"
    assert ref.context_scope_id == "ctx-x"
    assert ref.lifecycle_status == ReusableArtifactStatus.VALIDATED.value


class _CustomReadPort:
    async def read_references(
        self,
        identity: RequestIdentity,
        request: UclReferenceReadRequest,
    ) -> UclReferenceReadResult:
        return UclReferenceReadResult(outcome=UclReferenceReadOutcome.OK)


def test_custom_read_port_still_supported() -> None:
    port: UclReferenceReadPort = _CustomReadPort()
    assert isinstance(port, UclReferenceReadPort)


def test_anti_regression_no_workspace_equals_context_scope_rule() -> None:
    source = _DEFAULT_READER.read_text(encoding="utf-8")
    assert "workspace_id == context_scope_id" not in source
    assert "context_scope_id == workspace_id" not in source
