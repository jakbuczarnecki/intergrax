# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-B pinned tests: F-D2-02 / F-D2-14 TASK scope is tenant-qualified, guarded and governed."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from intergrax.applications._shared.memory_control_wiring import build_default_memory_control_plane
from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlForgetRequest,
    MemoryControlGovernanceDenied,
    MemoryControlNotFound,
    MemoryControlRememberRequest,
)
from tests.qualification.memory_behavior.fixtures import (
    InMemoryTaskMemoryCapability,
    deny_remember_governance,
    request_identity,
    task_scope,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _payload(value: str) -> MemoryControlRememberRequest:
    return MemoryControlRememberRequest(content=json.dumps({"v": value}))


def _values(store: InMemoryTaskMemoryCapability) -> list[str]:
    return sorted(str(v) for v in store._data.values())


@dataclass
class _CountingStore(InMemoryTaskMemoryCapability):
    calls: list[str] = field(default_factory=list)

    async def write(self, namespace: str, key: str, value: dict[str, object]) -> None:
        self.calls.append("write")
        await super().write(namespace, key, value)

    async def delete(self, namespace: str, key: str) -> bool:
        self.calls.append("delete")
        return await super().delete(namespace, key)


@dataclass
class _TenantBoundStore(_CountingStore):
    """Duck-typed capability that exposes the tenant it is bound to (not the shipped wrapper)."""

    tenant: str = "tenant-bound"

    @property
    def bound_tenant_id(self) -> str:
        return self.tenant


# --------------------------------------------------------------------------- isolation


@pytest.mark.asyncio
async def test_f_d2_02_two_tenants_do_not_overwrite_each_others_task_memory() -> None:
    store = InMemoryTaskMemoryCapability()
    plane = build_default_memory_control_plane(task_memory=store)
    a = request_identity(tenant_id="tenant-a", user_id="u")
    b = request_identity(tenant_id="tenant-b", user_id="u")

    await plane.remember(a, task_scope(a, namespace="ns", key="k"), _payload("a"))
    await plane.remember(b, task_scope(b, namespace="ns", key="k"), _payload("b"))

    assert len(store._data) == 2
    assert _values(store) == ["{'v': 'a'}", "{'v': 'b'}"]


@pytest.mark.asyncio
async def test_f_d2_02_a_tenant_cannot_forget_another_tenants_key() -> None:
    store = InMemoryTaskMemoryCapability()
    plane = build_default_memory_control_plane(task_memory=store)
    a = request_identity(tenant_id="tenant-a", user_id="u")
    c = request_identity(tenant_id="tenant-c", user_id="u")
    await plane.remember(a, task_scope(a, namespace="ns", key="k"), _payload("a"))

    with pytest.raises(MemoryControlNotFound):
        await plane.forget(c, task_scope(c, namespace="ns", key="k"), MemoryControlForgetRequest())

    assert _values(store) == ["{'v': 'a'}"]


@pytest.mark.asyncio
async def test_f_d2_02_forget_removes_only_the_requesting_tenants_entry() -> None:
    store = InMemoryTaskMemoryCapability()
    plane = build_default_memory_control_plane(task_memory=store)
    a = request_identity(tenant_id="tenant-a", user_id="u")
    b = request_identity(tenant_id="tenant-b", user_id="u")
    await plane.remember(a, task_scope(a, namespace="ns", key="k"), _payload("a"))
    await plane.remember(b, task_scope(b, namespace="ns", key="k"), _payload("b"))

    await plane.forget(a, task_scope(a, namespace="ns", key="k"), MemoryControlForgetRequest())

    assert _values(store) == ["{'v': 'b'}"]


@pytest.mark.asyncio
async def test_f_d2_02_tenant_qualification_is_injective() -> None:
    """Separator characters in a tenant id or namespace cannot make two pairs collide."""
    store = InMemoryTaskMemoryCapability()
    plane = build_default_memory_control_plane(task_memory=store)
    pairs = [("a", "b/c"), ("a/b", "c"), ("a:b", "c"), ("a", "b:c"), ("a", "b"), ("ab", "c")]
    for tenant, namespace in pairs:
        identity = request_identity(tenant_id=tenant, user_id="u")
        await plane.remember(identity, task_scope(identity, namespace=namespace, key="k"), _payload(tenant + namespace))

    assert len(store._data) == len(pairs)


# --------------------------------------------------------------------------- governance and guard


@pytest.mark.asyncio
async def test_f_d2_02_task_remember_and_forget_run_the_governance_policy() -> None:
    store = _CountingStore()
    plane = build_default_memory_control_plane(task_memory=store, security_governance=deny_remember_governance())
    identity = request_identity(tenant_id="tenant-a", user_id="u")
    scope = task_scope(identity, namespace="ns", key="k")

    with pytest.raises(MemoryControlGovernanceDenied):
        await plane.remember(identity, scope, _payload("a"))
    with pytest.raises(MemoryControlGovernanceDenied):
        await plane.forget(identity, scope, MemoryControlForgetRequest())

    assert store.calls == []
    assert store._data == {}


@pytest.mark.asyncio
async def test_f_d2_02_default_governance_still_admits_task_memory() -> None:
    store = _CountingStore()
    plane = build_default_memory_control_plane(task_memory=store)
    identity = request_identity(tenant_id="tenant-a", user_id="u")
    scope = task_scope(identity, namespace="ns", key="k")

    result = await plane.remember(identity, scope, _payload("a"))
    deleted = await plane.forget(identity, scope, MemoryControlForgetRequest())

    assert result.task_written is True
    assert deleted.task_deleted is True
    assert store.calls == ["write", "delete"]


@pytest.mark.asyncio
async def test_f_d2_14_a_duck_typed_capability_exposing_its_tenant_is_guarded_like_the_wrapper() -> None:
    store = _TenantBoundStore(tenant="tenant-a")
    plane = build_default_memory_control_plane(task_memory=store)
    foreign = request_identity(tenant_id="tenant-b", user_id="u")
    own = request_identity(tenant_id="tenant-a", user_id="u")

    with pytest.raises(MemoryControlAccessDenied):
        await plane.remember(foreign, task_scope(foreign, namespace="ns", key="k"), _payload("x"))
    with pytest.raises(MemoryControlAccessDenied):
        await plane.forget(foreign, task_scope(foreign, namespace="ns", key="k"), MemoryControlForgetRequest())
    assert store.calls == []

    await plane.remember(own, task_scope(own, namespace="ns", key="k"), _payload("ok"))
    assert store.calls == ["write"]


@pytest.mark.asyncio
async def test_f_d2_02_scope_tenant_spoof_is_still_denied_before_any_capability_call() -> None:
    from intergrax.memory.contracts.memory_control import MemoryControlPlaneScope, MemoryControlScopeRef

    store = _CountingStore()
    plane = build_default_memory_control_plane(task_memory=store)
    identity = request_identity(tenant_id="tenant-a", user_id="u")
    spoofed = MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.TASK, tenant_id="tenant-b", task_namespace="ns", task_key="k"
    )
    with pytest.raises(MemoryControlAccessDenied):
        await plane.remember(identity, spoofed, _payload("x"))
    assert store.calls == []
