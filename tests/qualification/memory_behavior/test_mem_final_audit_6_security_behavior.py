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
from tests.qualification.memory_behavior.fixtures import (
    TENANT_A,
    TENANT_B,
    build_user_control_plane,
    request_identity,
)

pytestmark = pytest.mark.gate


@pytest.mark.asyncio
async def test_identity_user_spoof_denied() -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id="user-a")
    scope = MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.USER,
        tenant_id=TENANT_A,
        user_id="user-b",
    )
    with pytest.raises(MemoryControlAccessDenied):
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="spoof"),
        )
    with pytest.raises(MemoryControlAccessDenied):
        await plane.recall(identity, scope, MemoryControlRecallRequest(top_k=3))
    with pytest.raises(MemoryControlAccessDenied):
        await plane.forget(identity, scope, MemoryControlForgetRequest(entry_id="x"))


@pytest.mark.asyncio
async def test_identity_tenant_spoof_denied() -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(tenant_id=TENANT_A, user_id="user-a")
    scope = MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.USER,
        tenant_id=TENANT_B,
        user_id="user-a",
    )
    with pytest.raises(MemoryControlAccessDenied):
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="tenant spoof"),
        )


@pytest.mark.asyncio
async def test_cross_user_isolation() -> None:
    plane, _ = build_user_control_plane()
    user_a = request_identity(user_id="user-a")
    user_b = request_identity(user_id="user-b")
    scope_a = user_memory_scope(user_a)
    scope_b = user_memory_scope(user_b)
    remembered = await plane.remember(
        user_a,
        scope_a,
        MemoryControlRememberRequest(content="secret-A"),
    )
    recall_b = await plane.recall(
        user_b,
        scope_b,
        MemoryControlRecallRequest(query="secret", top_k=10),
    )
    assert all(item.entry_id != remembered.entry_id for item in recall_b.items)
    assert all("secret-A" not in item.content for item in recall_b.items)


@pytest.mark.asyncio
async def test_cross_tenant_isolation() -> None:
    plane_a, _ = build_user_control_plane(tenant_id=TENANT_A)
    plane_b, _ = build_user_control_plane(tenant_id=TENANT_B)
    shared_user = "shared-user-id"
    identity_a = request_identity(tenant_id=TENANT_A, user_id=shared_user)
    identity_b = request_identity(tenant_id=TENANT_B, user_id=shared_user)
    scope_a = user_memory_scope(identity_a)
    scope_b = user_memory_scope(identity_b)
    remembered = await plane_a.remember(
        identity_a,
        scope_a,
        MemoryControlRememberRequest(content="tenant-secret"),
    )
    recall_b = await plane_b.recall(
        identity_b,
        scope_b,
        MemoryControlRecallRequest(query="tenant", top_k=10),
    )
    assert all(item.entry_id != remembered.entry_id for item in recall_b.items)
