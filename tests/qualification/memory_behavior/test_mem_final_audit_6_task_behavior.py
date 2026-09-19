# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6 — TASK scope via MemoryControlPlane."""

from __future__ import annotations

import json

import pytest

from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlForgetRequest,
    MemoryControlPlaneScope,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    MemoryControlScopeRef,
    MemoryControlUnsupportedScope,
)
from tests.qualification.memory_behavior.fixtures import (
    InMemoryTaskMemoryCapability,
    TENANT_A,
    TENANT_B,
    build_task_control_plane,
    request_identity,
    task_scope,
)

pytestmark = pytest.mark.gate


@pytest.mark.asyncio
async def test_task_01_remember_recall() -> None:
    store = InMemoryTaskMemoryCapability()
    plane = build_task_control_plane(store)
    identity = request_identity(user_id="task-user")
    scope = task_scope(identity, namespace="ns-a", key="checkpoint")
    await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(task_value_json=json.dumps({"step": 1})),
    )
    with pytest.raises(MemoryControlUnsupportedScope):
        await plane.recall(
            identity,
            scope,
            MemoryControlRecallRequest(query="", top_k=1),
        )
    value = await store.read("ns-a", "checkpoint")
    assert value == {"step": 1}


@pytest.mark.asyncio
async def test_task_02_forget() -> None:
    store = InMemoryTaskMemoryCapability()
    plane = build_task_control_plane(store)
    identity = request_identity(user_id="task-user")
    scope = task_scope(identity, namespace="ns-a", key="temp")
    await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content=json.dumps({"x": 1})),
    )
    await plane.forget(identity, scope, MemoryControlForgetRequest())
    assert await store.read("ns-a", "temp") is None


@pytest.mark.asyncio
async def test_task_03_namespace_isolation() -> None:
    store = InMemoryTaskMemoryCapability()
    plane = build_task_control_plane(store)
    identity = request_identity(user_id="task-user")
    scope_a = task_scope(identity, namespace="ns-a", key="same-key")
    scope_b = task_scope(identity, namespace="ns-b", key="same-key")
    await plane.remember(
        identity,
        scope_a,
        MemoryControlRememberRequest(content=json.dumps({"from": "a"})),
    )
    await plane.remember(
        identity,
        scope_b,
        MemoryControlRememberRequest(content=json.dumps({"from": "b"})),
    )
    assert await store.read("ns-a", "same-key") == {"from": "a"}
    assert await store.read("ns-b", "same-key") == {"from": "b"}


@pytest.mark.asyncio
async def test_task_05_tenant_scope_on_task() -> None:
    store = InMemoryTaskMemoryCapability()
    plane = build_task_control_plane(store)
    identity = request_identity(tenant_id=TENANT_A, user_id="task-user")
    scope = MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.TASK,
        tenant_id=TENANT_B,
        task_namespace="ns",
        task_key="k",
    )
    with pytest.raises(MemoryControlAccessDenied):
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content=json.dumps({"bad": True})),
        )
