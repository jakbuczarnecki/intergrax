# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.task_memory import InMemoryTaskMemoryStore, MemoryAccessPolicy, PolicyScopedMemoryView
from intergrax.runtime.task_memory.memory_view import MemoryViewAccessDenied
from testing_support.builder import build_runtime_execution_context_for_tests

pytestmark = pytest.mark.gate

_CANONICAL_TENANT = "tenant-a"
_CONFLICT_TENANT = "tenant-b"


def _view_with_identity(
    *,
    metadata: dict[str, object] | None = None,
    tenant_id: str = _CANONICAL_TENANT,
) -> PolicyScopedMemoryView:
    exec_ctx = build_runtime_execution_context_for_tests(
        tenant_id=tenant_id,
        metadata=metadata or {},
    )
    return PolicyScopedMemoryView(
        exec_ctx,
        InMemoryTaskMemoryStore(),
        access_policy=MemoryAccessPolicy(scope_boundary="tenant"),
    )


@pytest.mark.asyncio
async def test_metadata_tenant_conflict_rejects_read_write_list_delete() -> None:
    exec_ctx = build_runtime_execution_context_for_tests(tenant_id=_CANONICAL_TENANT)
    view = PolicyScopedMemoryView(
        exec_ctx,
        InMemoryTaskMemoryStore(),
        access_policy=MemoryAccessPolicy(scope_boundary="tenant"),
    )
    await view.write("ns", "k", {"v": 1})
    exec_ctx.metadata["tenant_id"] = _CONFLICT_TENANT
    with pytest.raises(MemoryViewAccessDenied, match="conflicts"):
        await view.read("ns", "k")
    with pytest.raises(MemoryViewAccessDenied, match="conflicts"):
        await view.write("ns", "k", {"v": 1})
    with pytest.raises(MemoryViewAccessDenied, match="conflicts"):
        await view.list("ns")
    with pytest.raises(MemoryViewAccessDenied, match="conflicts"):
        await view.delete("ns", "k")


@pytest.mark.asyncio
async def test_memory_scope_tenant_metadata_is_not_authority() -> None:
    exec_ctx = build_runtime_execution_context_for_tests(tenant_id=_CANONICAL_TENANT)
    view = PolicyScopedMemoryView(
        exec_ctx,
        InMemoryTaskMemoryStore(),
        access_policy=MemoryAccessPolicy(scope_boundary="tenant"),
    )
    exec_ctx.metadata["memory_scope_tenant_id"] = _CONFLICT_TENANT
    with pytest.raises(MemoryViewAccessDenied, match="not authority"):
        await view.read("ns", "k")


def test_memory_view_requires_canonical_identity_on_context() -> None:
    exec_ctx = build_runtime_execution_context_for_tests()
    exec_ctx.canonical_request_identity = None
    with pytest.raises(MemoryViewAccessDenied, match="canonical request identity"):
        PolicyScopedMemoryView(exec_ctx, InMemoryTaskMemoryStore())


def test_task_scope_uses_execution_context_task_id_only() -> None:
    exec_ctx = build_runtime_execution_context_for_tests(tenant_id=_CANONICAL_TENANT)
    view = PolicyScopedMemoryView(exec_ctx, InMemoryTaskMemoryStore())
    assert view._task_id == str(exec_ctx.task_id)
