# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6 — identity and isolation hard gates."""

from __future__ import annotations

import pytest

from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlForgetRequest,
    MemoryControlPlaneScope,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    MemoryControlScopeRef,
    user_memory_scope,
)
from tests.qualification.memory_behavior.contracts import BehaviorViolationLedger
from tests.qualification.memory_behavior.fixtures import (
    TENANT_A,
    TENANT_B,
    build_shared_user_control_planes_for_tenants,
    build_user_control_plane,
    request_identity,
)
from tests.qualification.memory_behavior.gate_helpers import (
    assert_recall_does_not_expose_content,
    assert_recall_does_not_expose_entry,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore

pytestmark = pytest.mark.gate


@pytest.mark.asyncio
async def test_identity_user_spoof_denied(violation_ledger: BehaviorViolationLedger) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id="user-a")
    scope = MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.USER,
        tenant_id=TENANT_A,
        user_id="user-b",
    )
    try:
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="spoof"),
        )
        violation_ledger.record_identity_violation()
        raise AssertionError("remember must deny user scope spoof")
    except MemoryControlAccessDenied:
        pass
    try:
        await plane.recall(identity, scope, MemoryControlRecallRequest(top_k=3))
        violation_ledger.record_identity_violation()
        raise AssertionError("recall must deny user scope spoof")
    except MemoryControlAccessDenied:
        pass
    try:
        await plane.forget(identity, scope, MemoryControlForgetRequest(entry_id="x"))
        violation_ledger.record_identity_violation()
        raise AssertionError("forget must deny user scope spoof")
    except MemoryControlAccessDenied:
        pass


@pytest.mark.asyncio
async def test_identity_tenant_spoof_denied(violation_ledger: BehaviorViolationLedger) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(tenant_id=TENANT_A, user_id="user-a")
    scope = MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.USER,
        tenant_id=TENANT_B,
        user_id="user-a",
    )
    try:
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="tenant spoof"),
        )
        violation_ledger.record_identity_violation()
        raise AssertionError("remember must deny tenant scope spoof")
    except MemoryControlAccessDenied:
        pass


@pytest.mark.asyncio
async def test_cross_user_isolation(violation_ledger: BehaviorViolationLedger) -> None:
    shared_store = InMemoryUserProfileStore()
    plane, _ = build_user_control_plane(store=shared_store)
    user_a = request_identity(user_id="user-a")
    user_b = request_identity(user_id="user-b")
    scope_a = user_memory_scope(user_a)
    scope_b = user_memory_scope(user_b)
    remembered = await plane.remember(
        user_a,
        scope_a,
        MemoryControlRememberRequest(content="secret-A"),
    )
    assert remembered.entry_id
    recall_b = await plane.recall(
        user_b,
        scope_b,
        MemoryControlRecallRequest(query="secret", top_k=10),
    )
    assert_recall_does_not_expose_entry(
        recall_b.items,
        forbidden_entry_id=remembered.entry_id,
        ledger=violation_ledger,
        leak_kind="cross_user",
    )
    assert_recall_does_not_expose_content(
        recall_b.items,
        forbidden_substring="secret-A",
        ledger=violation_ledger,
        leak_kind="cross_user",
    )


@pytest.mark.asyncio
async def test_cross_tenant_isolation(violation_ledger: BehaviorViolationLedger) -> None:
    shared_store, plane_a, plane_b, _manager_a, _manager_b = build_shared_user_control_planes_for_tenants()
    assert shared_store is not None
    shared_user = "shared-user-id"
    identity_a = request_identity(tenant_id=TENANT_A, user_id=shared_user)
    identity_b = request_identity(tenant_id=TENANT_B, user_id=shared_user)
    scope_a = user_memory_scope(identity_a)
    scope_b = user_memory_scope(identity_b)
    secret_a = await plane_a.remember(
        identity_a,
        scope_a,
        MemoryControlRememberRequest(content="tenant-secret-A"),
    )
    secret_b = await plane_b.remember(
        identity_b,
        scope_b,
        MemoryControlRememberRequest(content="tenant-secret-B"),
    )
    assert secret_a.entry_id and secret_b.entry_id
    recall_b = await plane_b.recall(
        identity_b,
        scope_b,
        MemoryControlRecallRequest(query="tenant", top_k=10),
    )
    assert_recall_does_not_expose_entry(
        recall_b.items,
        forbidden_entry_id=secret_a.entry_id,
        ledger=violation_ledger,
        leak_kind="cross_tenant",
    )
    assert_recall_does_not_expose_content(
        recall_b.items,
        forbidden_substring="tenant-secret-A",
        ledger=violation_ledger,
        leak_kind="cross_tenant",
    )
    assert any(item.entry_id == secret_b.entry_id for item in recall_b.items)


@pytest.mark.asyncio
async def test_cross_tenant_shared_backing_reverse_direction(
    violation_ledger: BehaviorViolationLedger,
) -> None:
    _shared_store, plane_a, plane_b, _manager_a, _manager_b = (
        build_shared_user_control_planes_for_tenants()
    )
    shared_user = "shared-user-id"
    identity_a = request_identity(tenant_id=TENANT_A, user_id=shared_user)
    identity_b = request_identity(tenant_id=TENANT_B, user_id=shared_user)
    scope_a = user_memory_scope(identity_a)
    scope_b = user_memory_scope(identity_b)
    await plane_a.remember(
        identity_a,
        scope_a,
        MemoryControlRememberRequest(content="tenant-secret-A"),
    )
    secret_b = await plane_b.remember(
        identity_b,
        scope_b,
        MemoryControlRememberRequest(content="tenant-secret-B"),
    )
    assert secret_b.entry_id
    recall_a = await plane_a.recall(
        identity_a,
        scope_a,
        MemoryControlRecallRequest(query="tenant", top_k=10),
    )
    assert_recall_does_not_expose_entry(
        recall_a.items,
        forbidden_entry_id=secret_b.entry_id,
        ledger=violation_ledger,
        leak_kind="cross_tenant",
    )
    assert_recall_does_not_expose_content(
        recall_a.items,
        forbidden_substring="tenant-secret-B",
        ledger=violation_ledger,
        leak_kind="cross_tenant",
    )
