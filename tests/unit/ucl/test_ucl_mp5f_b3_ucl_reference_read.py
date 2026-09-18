# © Artur Czarnecki. All rights reserved.

"""MP-5F-B3: UCL scoped reference-read contract, isolation, and pluginability."""

from __future__ import annotations

import ast
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.runtime.context_lifecycle import (
    ArtifactCompressionTarget,
    ArtifactLookupKey,
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    InMemoryOptimizationArtifactRepository,
    OptimizationArtifactType,
    ReusableArtifactStatus,
    ReusableOptimizationArtifact,
    StoredOptimizationArtifact,
    build_optimization_artifact_reference,
    compute_artifact_content_hash,
)
from intergrax.runtime.context_lifecycle.default_ucl_reference_reader import (
    DefaultUclReferenceReader,
    UclReferenceReadCapabilityBinding,
    UclReferenceReadConfigurationError,
)
from intergrax.ucl.contracts.ucl_reference_read import (
    UCL_REFERENCE_READ_MAX_LIMIT,
    UclOptimizationArtifactCanonicalRef,
    UclReferenceLifecycleSelection,
    UclReferenceReadOutcome,
    UclReferenceReadPort,
    UclReferenceReadQuery,
    UclReferenceReadRequest,
    UclReferenceReadResult,
    UclReferenceReadScope,
    UclReferenceReadScopeError,
    UclScopedResourceRef,
    format_ucl_artifact_locator,
    validate_ucl_reference_read_request,
)

pytestmark = pytest.mark.gate

_REPO = Path(__file__).resolve().parents[3]
_CONTRACT = _REPO / "intergrax" / "ucl" / "contracts" / "ucl_reference_read.py"
_DEFAULT_READER = (
    _REPO
    / "intergrax"
    / "runtime"
    / "context_lifecycle"
    / "default_ucl_reference_reader.py"
)

_BASE_TIME = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)

_FORBIDDEN_CONTRACT_IMPORT_PREFIXES = (
    "intergrax.contracts.context_view",
    "intergrax.contracts.context_view_source_ports",
    "intergrax.contracts.context_view_composition",
    "intergrax.collaborative_work",
    "intergrax.memory",
    "intergrax.rag",
    "intergrax.runtime.nexus",
)

_FORBIDDEN_DEFAULT_IMPORT_PREFIXES = _FORBIDDEN_CONTRACT_IMPORT_PREFIXES + (
    "intergrax.runtime.context_lifecycle.sqlite_repository",
    "intergrax.runtime.token_optimization",
    "intergrax.context",
)


def _identity(
    *,
    tenant_id: str = "tenant-a",
    user_id: str = "user-1",
) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def _scope(
    *,
    tenant_id: str = "tenant-a",
    context_scope_id: str = "ws-a",
    workspace_id: str | None = "ws-a",
) -> UclReferenceReadScope:
    return UclReferenceReadScope(
        tenant_id=tenant_id,
        context_scope_id=context_scope_id,
        workspace_id=workspace_id,
    )


def _lookup_key(**overrides: object) -> ArtifactLookupKey:
    defaults: dict[str, object] = {
        "tenant_id": "tenant-a",
        "context_scope_id": "ws-a",
        "artifact_type": OptimizationArtifactType.MESSAGE_SEQUENCE,
        "source_content_hash": "hash-abc",
        "strategy_id": "strategy.summarize",
        "strategy_version": "1.0.0",
        "policy_version": "policy-v1",
        "validation_contract_version": "validation-v1",
        "compression_target": ArtifactCompressionTarget(target_tokens=1000),
        "lossiness_profile": "lossy_summary",
        "source_refs": ("msg-1", "msg-2"),
    }
    defaults.update(overrides)
    return ArtifactLookupKey(**defaults)  # type: ignore[arg-type]


def _stored(
    *,
    artifact_id: str = "artifact-1",
    payload: bytes = b"payload-bytes",
    lookup_key: ArtifactLookupKey | None = None,
    status: ReusableArtifactStatus = ReusableArtifactStatus.VALIDATED,
) -> StoredOptimizationArtifact:
    key = lookup_key or _lookup_key()
    metadata = ReusableOptimizationArtifact(
        artifact_id=artifact_id,
        lookup_key=key,
        artifact_content_hash=compute_artifact_content_hash(payload),
        created_at=_BASE_TIME,
        created_by_executor="executor.message_sequence",
        validation=ArtifactValidationSummary(
            status=ArtifactValidationStatus.PASSED,
            validation_contract_version="validation-v1",
            validated_at=_BASE_TIME,
        ),
        status=status,
    )
    return StoredOptimizationArtifact(
        metadata=metadata,
        payload=payload,
        media_type="application/octet-stream",
    )


def _seed_repo(*artifacts: StoredOptimizationArtifact) -> InMemoryOptimizationArtifactRepository:
    repo = InMemoryOptimizationArtifactRepository()
    for artifact in artifacts:
        key = artifact.metadata.lookup_key
        reservation = repo.try_acquire_creation_reservation(
            key,
            owner_operation_id="op-seed",
            lease_seconds=60,
        )
        assert reservation.reservation is not None
        repo.store_validated_artifact(
            reservation=reservation.reservation,
            artifact=artifact,
        )
    return repo


def _reader(
    repo: InMemoryOptimizationArtifactRepository,
    *,
    tenant_id: str = "tenant-a",
    context_scope_id: str = "ws-a",
) -> DefaultUclReferenceReader:
    return DefaultUclReferenceReader(
        catalog=repo,
        capability_binding=UclReferenceReadCapabilityBinding(
            tenant_id=tenant_id,
            context_scope_id=context_scope_id,
        ),
    )


def test_contract_request_immutability_and_query_bounds() -> None:
    _ = UclReferenceReadRequest(scope=_scope())
    with pytest.raises(UclReferenceReadScopeError):
        UclReferenceReadQuery(limit=0)
    with pytest.raises(UclReferenceReadScopeError):
        UclReferenceReadQuery(limit=UCL_REFERENCE_READ_MAX_LIMIT + 1)
    with pytest.raises(UclReferenceReadScopeError):
        UclReferenceReadScope(tenant_id="", context_scope_id="ws")
    with pytest.raises(UclReferenceReadScopeError):
        UclReferenceReadScope(tenant_id="t", context_scope_id="")
    with pytest.raises(UclReferenceReadScopeError):
        UclReferenceReadScope(
            tenant_id="t",
            context_scope_id="ws-a",
            workspace_id="ws-b",
        )


def test_validate_identity_tenant_mismatch_scope_rejected() -> None:
    identity = _identity(tenant_id="tenant-a")
    request = UclReferenceReadRequest(scope=_scope(tenant_id="tenant-b"))
    assert (
        validate_ucl_reference_read_request(identity, request)
        is UclReferenceReadOutcome.SCOPE_REJECTED
    )


def test_result_reference_only_no_payload_fields() -> None:
    ref = UclOptimizationArtifactCanonicalRef(
        tenant_id="tenant-a",
        context_scope_id="ws-a",
        artifact_id="art-1",
        artifact_lookup_key_hash="key-hash",
        artifact_content_hash="content-hash",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE.value,
        lifecycle_status=ReusableArtifactStatus.VALIDATED.value,
    )
    payload = dataclasses.asdict(ref)
    forbidden = {"content", "summary", "payload", "text", "body", "prompt", "messages"}
    assert forbidden.isdisjoint(payload.keys())
    locator = format_ucl_artifact_locator(ref)
    assert ref.artifact_id in locator
    assert ref.artifact_lookup_key_hash in locator


def test_canonical_identity_requires_artifact_and_lookup_hash() -> None:
    with pytest.raises(UclReferenceReadScopeError):
        UclOptimizationArtifactCanonicalRef(
            tenant_id="t",
            context_scope_id="s",
            artifact_id="",
            artifact_lookup_key_hash="h",
            artifact_content_hash="c",
            artifact_type="message_sequence",
            lifecycle_status="validated",
        )
    with pytest.raises(UclReferenceReadScopeError):
        UclOptimizationArtifactCanonicalRef(
            tenant_id="t",
            context_scope_id="s",
            artifact_id="a",
            artifact_lookup_key_hash="",
            artifact_content_hash="c",
            artifact_type="message_sequence",
            lifecycle_status="validated",
        )


def test_non_ok_result_cannot_carry_references() -> None:
    ref = UclOptimizationArtifactCanonicalRef(
        tenant_id="t",
        context_scope_id="s",
        artifact_id="a",
        artifact_lookup_key_hash="k",
        artifact_content_hash="c",
        artifact_type="message_sequence",
        lifecycle_status="validated",
    )
    with pytest.raises(UclReferenceReadScopeError):
        UclReferenceReadResult(
            outcome=UclReferenceReadOutcome.ACCESS_DENIED,
            references=(ref,),
        )


def test_configured_reader_requires_capability_binding() -> None:
    repo = InMemoryOptimizationArtifactRepository()
    with pytest.raises(UclReferenceReadConfigurationError):
        DefaultUclReferenceReader(catalog=repo)


@pytest.mark.asyncio
async def test_empty_ok_when_no_artifacts() -> None:
    repo = InMemoryOptimizationArtifactRepository()
    reader = _reader(repo)
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(scope=_scope()),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert result.references == ()


@pytest.mark.asyncio
async def test_default_reader_returns_refs_without_payload() -> None:
    stored = _stored()
    repo = _seed_repo(stored)
    reader = _reader(repo)
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(scope=_scope()),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert len(result.references) == 1
    ref = result.references[0]
    assert ref.artifact_id == stored.metadata.artifact_id
    assert ref.artifact_lookup_key_hash
    assert "payload" not in dataclasses.asdict(ref)


@pytest.mark.asyncio
async def test_cross_tenant_scope_rejected() -> None:
    reader = DefaultUclReferenceReader()
    result = await reader.read_references(
        _identity(tenant_id="tenant-a"),
        UclReferenceReadRequest(scope=_scope(tenant_id="tenant-b")),
    )
    assert result.outcome is UclReferenceReadOutcome.SCOPE_REJECTED


@pytest.mark.asyncio
async def test_wrong_context_scope_rejected_via_binding() -> None:
    repo = _seed_repo(_stored())
    reader = _reader(repo, context_scope_id="ws-other")
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(scope=_scope(context_scope_id="ws-a")),
    )
    assert result.outcome is UclReferenceReadOutcome.SCOPE_REJECTED


@pytest.mark.asyncio
async def test_artifact_in_other_scope_not_returned() -> None:
    stored = _stored(lookup_key=_lookup_key(context_scope_id="ws-b"))
    repo = _seed_repo(stored)
    reader = _reader(repo, context_scope_id="ws-a")
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(scope=_scope(context_scope_id="ws-a")),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert result.references == ()


@pytest.mark.asyncio
async def test_resource_scope_mismatch_filters_to_empty() -> None:
    stored = _stored()
    repo = _seed_repo(stored)
    reader = _reader(repo)
    scope = UclReferenceReadScope(
        tenant_id="tenant-a",
        context_scope_id="ws-a",
        workspace_id="ws-a",
        resource=UclScopedResourceRef(resource_kind="source_ref", resource_id="missing"),
    )
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(scope=scope),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert result.references == ()


@pytest.mark.asyncio
async def test_unsupported_resource_kind_rejected() -> None:
    repo = _seed_repo(_stored())
    reader = _reader(repo)
    scope = UclReferenceReadScope(
        tenant_id="tenant-a",
        context_scope_id="ws-a",
        workspace_id="ws-a",
        resource=UclScopedResourceRef(resource_kind="work_item", resource_id="wi-1"),
    )
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(scope=scope),
    )
    assert result.outcome is UclReferenceReadOutcome.SCOPE_REJECTED


@pytest.mark.asyncio
async def test_active_only_excludes_invalidated_historical() -> None:
    stored = _stored(artifact_id="artifact-1")
    repo = _seed_repo(stored)
    reference = build_optimization_artifact_reference(stored)
    invalidated = repo.invalidate_artifact(reference, reason="superseded")
    assert invalidated is not None
    reader = _reader(repo)
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=_scope(),
            query=UclReferenceReadQuery(
                lifecycle_selection=UclReferenceLifecycleSelection.ACTIVE_VALIDATED_ONLY,
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert result.references == ()


@pytest.mark.asyncio
async def test_include_historical_returns_invalidated_row() -> None:
    stored = _stored(artifact_id="artifact-hist")
    repo = _seed_repo(stored)
    reference = build_optimization_artifact_reference(stored)
    repo.invalidate_artifact(reference, reason="superseded")
    reader = _reader(repo)
    result = await reader.read_references(
        _identity(),
        UclReferenceReadRequest(
            scope=_scope(),
            query=UclReferenceReadQuery(
                lifecycle_selection=UclReferenceLifecycleSelection.INCLUDE_HISTORICAL,
            ),
        ),
    )
    assert result.outcome is UclReferenceReadOutcome.OK
    assert len(result.references) == 1
    assert result.references[0].lifecycle_status == ReusableArtifactStatus.INVALIDATED.value


class _CustomUclReferenceReader:
    async def read_references(
        self,
        identity: RequestIdentity,
        request: UclReferenceReadRequest,
    ) -> UclReferenceReadResult:
        if request.scope.tenant_id != identity.tenant_id:
            return UclReferenceReadResult(
                outcome=UclReferenceReadOutcome.SCOPE_REJECTED,
            )
        ref = UclOptimizationArtifactCanonicalRef(
            tenant_id=request.scope.tenant_id,
            context_scope_id=request.scope.context_scope_id,
            artifact_id="custom-artifact",
            artifact_lookup_key_hash="custom-key",
            artifact_content_hash="custom-content",
            artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE.value,
            lifecycle_status=ReusableArtifactStatus.VALIDATED.value,
        )
        return UclReferenceReadResult(
            outcome=UclReferenceReadOutcome.OK,
            references=(ref,),
        )


def test_pluginability_custom_port_without_default_runtime() -> None:
    port: UclReferenceReadPort = _CustomUclReferenceReader()
    assert isinstance(port, UclReferenceReadPort)


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
