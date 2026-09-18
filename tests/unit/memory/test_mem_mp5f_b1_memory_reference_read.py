# © Artur Czarnecki. All rights reserved.

"""MP-5F-B1: Memory scoped reference-read contract, isolation, and pluginability."""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.memory.contracts.memory_reference_read import (
    MEMORY_REFERENCE_READ_MAX_LIMIT,
    MemoryRecordCanonicalRef,
    MemoryReferenceReadOutcome,
    MemoryReferenceReadPort,
    MemoryReferenceReadQuery,
    MemoryReferenceReadRequest,
    MemoryReferenceReadResult,
    MemoryReferenceReadScope,
    MemoryReferenceReadScopeError,
    MemoryScopedResourceRef,
    validate_memory_reference_read_request,
)
from intergrax.memory.default_memory_reference_reader import (
    DefaultMemoryReferenceReader,
    MemoryReferenceReadCapabilityBinding,
    MemoryReferenceReadConfigurationError,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.default_memory_control_plane import UserProfileManagerMemoryCapability
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry

pytestmark = pytest.mark.gate

_REPO = Path(__file__).resolve().parents[3]
_CONTRACT = _REPO / "intergrax" / "memory" / "contracts" / "memory_reference_read.py"
_DEFAULT_READER = (
    _REPO / "intergrax" / "memory" / "default_memory_reference_reader.py"
)

_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.contracts.context_view",
    "intergrax.contracts.context_view_source_ports",
    "intergrax.contracts.context_view_composition",
    "intergrax.collaborative_work",
    "intergrax.rag",
    "intergrax.ucl",
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
    workspace_id: str = "ws-a",
    user_id: str | None = "user-1",
) -> MemoryReferenceReadScope:
    return MemoryReferenceReadScope(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        user_id=user_id,
    )


def test_contract_request_immutability_and_query_bounds() -> None:
    scope = _scope()
    request = MemoryReferenceReadRequest(scope=scope)
    with pytest.raises(MemoryReferenceReadScopeError):
        MemoryReferenceReadQuery(limit=0)
    with pytest.raises(MemoryReferenceReadScopeError):
        MemoryReferenceReadQuery(limit=MEMORY_REFERENCE_READ_MAX_LIMIT + 1)
    with pytest.raises(MemoryReferenceReadScopeError):
        MemoryReferenceReadScope(tenant_id="", workspace_id="ws")
    with pytest.raises(MemoryReferenceReadScopeError):
        MemoryReferenceReadScope(tenant_id="t", workspace_id="")


def test_tenant_required_on_scope() -> None:
    with pytest.raises(MemoryReferenceReadScopeError):
        MemoryReferenceReadScope(tenant_id="  ", workspace_id="ws-a")


def test_validate_identity_tenant_mismatch_scope_rejected() -> None:
    identity = _identity(tenant_id="tenant-a")
    request = MemoryReferenceReadRequest(
        scope=_scope(tenant_id="tenant-b"),
    )
    assert (
        validate_memory_reference_read_request(identity, request)
        is MemoryReferenceReadOutcome.SCOPE_REJECTED
    )


def test_result_reference_only_no_payload_fields() -> None:
    ref = MemoryRecordCanonicalRef(
        tenant_id="tenant-a",
        memory_id="mem-1",
        revision=1,
    )
    payload = dataclasses.asdict(ref)
    forbidden = {"content", "summary", "embedding", "payload", "text", "body"}
    assert forbidden.isdisjoint(payload.keys())
    serialized = json.dumps(payload)
    for token in ("secret body", "embedding vector"):
        assert token not in serialized


def test_non_ok_result_cannot_carry_references() -> None:
    ref = MemoryRecordCanonicalRef(tenant_id="t", memory_id="m", revision=1)
    with pytest.raises(MemoryReferenceReadScopeError):
        MemoryReferenceReadResult(
            outcome=MemoryReferenceReadOutcome.ACCESS_DENIED,
            references=(ref,),
        )


def test_ok_result_requires_evaluated_scope() -> None:
    with pytest.raises(MemoryReferenceReadScopeError, match="evaluated_scope"):
        MemoryReferenceReadResult(outcome=MemoryReferenceReadOutcome.OK)


def _configured_reader(
    capability: UserProfileManagerMemoryCapability,
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "ws-a",
) -> DefaultMemoryReferenceReader:
    return DefaultMemoryReferenceReader(
        user_profile=capability,
        capability_binding=MemoryReferenceReadCapabilityBinding(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ),
    )


def test_configured_reader_requires_capability_binding() -> None:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store, tenant_id="tenant-a", workspace_id="ws-a")
    capability = UserProfileManagerMemoryCapability(manager)
    with pytest.raises(MemoryReferenceReadConfigurationError):
        DefaultMemoryReferenceReader(user_profile=capability)


def test_capability_binding_requires_non_empty_workspace() -> None:
    with pytest.raises(MemoryReferenceReadConfigurationError):
        MemoryReferenceReadCapabilityBinding(
            tenant_id="tenant-a",
            workspace_id="  ",
        )


@pytest.mark.asyncio
async def test_empty_ok_when_no_records() -> None:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(
        store,
        tenant_id="tenant-a",
        workspace_id="ws-a",
    )
    capability = UserProfileManagerMemoryCapability(manager)
    reader = _configured_reader(capability)
    identity = _identity()
    result = await reader.read_references(
        identity,
        MemoryReferenceReadRequest(scope=_scope()),
    )
    assert result.outcome is MemoryReferenceReadOutcome.OK
    assert result.references == ()


@pytest.mark.asyncio
async def test_default_reader_returns_refs_without_content() -> None:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(
        store,
        tenant_id="tenant-a",
        workspace_id="ws-a",
    )
    capability = UserProfileManagerMemoryCapability(manager)
    entry = UserProfileMemoryEntry(
        content="secret preference text",
        kind=MemoryKind.PREFERENCE,
    )
    await capability.add_memory_entry(_identity(), entry)
    reader = _configured_reader(capability)
    result = await reader.read_references(
        _identity(),
        MemoryReferenceReadRequest(scope=_scope()),
    )
    assert result.outcome is MemoryReferenceReadOutcome.OK
    assert len(result.references) == 1
    ref = result.references[0]
    assert ref.memory_id == entry.entry_id
    assert "secret" not in dataclasses.asdict(ref).values()


@pytest.mark.asyncio
async def test_cross_tenant_scope_rejected() -> None:
    reader = DefaultMemoryReferenceReader()
    identity = _identity(tenant_id="tenant-a")
    request = MemoryReferenceReadRequest(
        scope=_scope(tenant_id="tenant-b"),
    )
    result = await reader.read_references(identity, request)
    assert result.outcome is MemoryReferenceReadOutcome.SCOPE_REJECTED


@pytest.mark.asyncio
async def test_correct_workspace_accepted_via_binding() -> None:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store, tenant_id="tenant-a", workspace_id="ws-a")
    capability = UserProfileManagerMemoryCapability(manager)
    reader = _configured_reader(capability)
    result = await reader.read_references(
        _identity(),
        MemoryReferenceReadRequest(scope=_scope(workspace_id="ws-a")),
    )
    assert result.outcome is MemoryReferenceReadOutcome.OK


@pytest.mark.asyncio
async def test_binding_tenant_mismatch_rejected() -> None:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store, tenant_id="tenant-a", workspace_id="ws-a")
    capability = UserProfileManagerMemoryCapability(manager)
    reader = _configured_reader(capability, tenant_id="tenant-b", workspace_id="ws-a")
    result = await reader.read_references(
        _identity(tenant_id="tenant-a"),
        MemoryReferenceReadRequest(scope=_scope(tenant_id="tenant-a")),
    )
    assert result.outcome is MemoryReferenceReadOutcome.SCOPE_REJECTED


@pytest.mark.asyncio
async def test_request_tenant_mismatch_rejected_before_binding() -> None:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store, tenant_id="tenant-b", workspace_id="ws-a")
    capability = UserProfileManagerMemoryCapability(manager)
    reader = _configured_reader(capability, tenant_id="tenant-b", workspace_id="ws-a")
    result = await reader.read_references(
        _identity(tenant_id="tenant-a"),
        MemoryReferenceReadRequest(scope=_scope(tenant_id="tenant-b")),
    )
    assert result.outcome is MemoryReferenceReadOutcome.SCOPE_REJECTED


@pytest.mark.asyncio
async def test_wrong_workspace_rejected_via_binding() -> None:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(
        store,
        tenant_id="tenant-a",
        workspace_id="ws-a",
    )
    capability = UserProfileManagerMemoryCapability(manager)
    reader = _configured_reader(capability)
    result = await reader.read_references(
        _identity(),
        MemoryReferenceReadRequest(scope=_scope(workspace_id="ws-other")),
    )
    assert result.outcome is MemoryReferenceReadOutcome.SCOPE_REJECTED


@pytest.mark.asyncio
async def test_resource_scope_rejected_when_unsupported() -> None:
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store, tenant_id="tenant-a", workspace_id="ws-a")
    capability = UserProfileManagerMemoryCapability(manager)
    reader = _configured_reader(capability)
    scope = MemoryReferenceReadScope(
        tenant_id="tenant-a",
        workspace_id="ws-a",
        user_id="user-1",
        resource=MemoryScopedResourceRef(
            resource_kind="work_item",
            resource_id="wi-1",
        ),
    )
    result = await reader.read_references(
        _identity(),
        MemoryReferenceReadRequest(scope=scope),
    )
    assert result.outcome is MemoryReferenceReadOutcome.SCOPE_REJECTED


class _CustomMemoryReferenceReader:
    """Pluginability proof — no default Memory runtime imports."""

    async def read_references(
        self,
        identity: RequestIdentity,
        request: MemoryReferenceReadRequest,
    ) -> MemoryReferenceReadResult:
        if request.scope.tenant_id != identity.tenant_id:
            return MemoryReferenceReadResult(
                outcome=MemoryReferenceReadOutcome.SCOPE_REJECTED,
            )
        ref = MemoryRecordCanonicalRef(
            tenant_id=request.scope.tenant_id,
            memory_id="custom-ref",
            revision=1,
        )
        return MemoryReferenceReadResult(
            outcome=MemoryReferenceReadOutcome.OK,
            references=(ref,),
            evaluated_scope=request.scope,
        )


@pytest.mark.asyncio
async def test_custom_reader_satisfies_port() -> None:
    port: MemoryReferenceReadPort = _CustomMemoryReferenceReader()
    result = await port.read_references(
        _identity(),
        MemoryReferenceReadRequest(scope=_scope()),
    )
    assert result.outcome is MemoryReferenceReadOutcome.OK
    assert result.references[0].memory_id == "custom-ref"


def _memory_reference_read_boundary_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        module = node.module
        for prefix in _FORBIDDEN_IMPORT_PREFIXES:
            if module.startswith(prefix):
                violations.append(f"{module} at line {node.lineno}")
        if module == "intergrax.memory.default_memory_control_plane":
            for alias in node.names:
                if alias.name.startswith("_"):
                    violations.append(
                        f"private control plane import {alias.name} at line {node.lineno}"
                    )
        if module == "typing" and any(
            alias.name == "Any" for alias in node.names
        ):
            violations.append(f"typing.Any at line {node.lineno}")
    source = path.read_text(encoding="utf-8")
    for name in ("getattr(", "hasattr(", "setattr("):
        if name in source:
            violations.append(f"dynamic attribute access: {name}")
    return violations


def test_memory_reference_read_contract_boundary_ast() -> None:
    violations = _memory_reference_read_boundary_violations(_CONTRACT)
    assert not violations, "\n".join(violations)


def test_memory_reference_read_default_impl_boundary_ast() -> None:
    violations = _memory_reference_read_boundary_violations(_DEFAULT_READER)
    assert not violations, "\n".join(violations)
