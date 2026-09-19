# © Artur Czarnecki. All rights reserved.

"""TASK scope behavioral scenarios."""

from __future__ import annotations

import json

from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlForgetRequest,
    MemoryControlPlaneScope,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    MemoryControlScopeRef,
    MemoryControlUnsupportedScope,
)
from tests.qualification.memory_behavior.contracts import BehaviorEvalContext
from tests.qualification.memory_behavior.fixtures import (
    InMemoryTaskMemoryCapability,
    TENANT_A,
    TENANT_B,
    build_task_control_plane,
    request_identity,
    task_scope,
)


async def run_task_01_remember_and_capability_read(_ctx: BehaviorEvalContext) -> None:
    store = InMemoryTaskMemoryCapability()
    plane = build_task_control_plane(store)
    identity = request_identity(user_id="task-user")
    scope = task_scope(identity, namespace="ns-a", key="checkpoint")
    await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(task_value_json=json.dumps({"step": 1})),
    )
    try:
        await plane.recall(
            identity,
            scope,
            MemoryControlRecallRequest(query="", top_k=1),
        )
    except MemoryControlUnsupportedScope:
        pass
    else:
        raise AssertionError("TASK recall via MemoryControlPlane must be unsupported")
    value = await store.read("ns-a", "checkpoint")
    assert value == {"step": 1}


async def run_task_02_forget(_ctx: BehaviorEvalContext) -> None:
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


async def run_task_03_namespace_isolation(_ctx: BehaviorEvalContext) -> None:
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


async def run_task_05_tenant_scope_on_task(_ctx: BehaviorEvalContext) -> None:
    store = InMemoryTaskMemoryCapability()
    plane = build_task_control_plane(store)
    identity = request_identity(tenant_id=TENANT_A, user_id="task-user")
    scope = MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.TASK,
        tenant_id=TENANT_B,
        task_namespace="ns",
        task_key="k",
    )
    try:
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content=json.dumps({"bad": True})),
        )
    except MemoryControlAccessDenied:
        return
    raise AssertionError("expected MemoryControlAccessDenied for tenant scope spoof on TASK")
