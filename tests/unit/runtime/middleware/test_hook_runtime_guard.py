# © Artur Czarnecki. All rights reserved.

"""APP-CON-5 — hook timeout, error→BLOCK, audit events."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.hooks.nexus_lifecycle_hooks import NexusLifecycleHookError
from intergrax.runtime.middleware.base import RuntimeMiddleware
from intergrax.runtime.middleware.hook_runtime_guard import invoke_guarded_hook
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class _SlowMiddleware(RuntimeMiddleware):
    priority = 60
    name = "slow_middleware"

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        await asyncio.sleep(0.05)
        return HookResult()

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult()


class _RaisingMiddleware(RuntimeMiddleware):
    priority = 61
    name = "raising_middleware"

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        raise RuntimeError("boom")

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult()


class _BlockingMiddleware(RuntimeMiddleware):
    priority = 62
    name = "blocking_middleware"

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult(action=HookAction.BLOCK, reason="policy_denied")

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult()


def _ctx() -> HookContext:
    return HookContext(
        task_id="task-1",
        run_id="run-1",
        phase=ExecutionPhase.INTAKE,
        runtime_state={},
    )


@pytest.mark.asyncio
async def test_invoke_guarded_hook_maps_exception_to_hook_error() -> None:
    async def _boom() -> HookResult:
        raise ValueError("middleware failed")

    result = await invoke_guarded_hook(
        hook_name="demo",
        point=HookPoint.BEFORE_TASK_INTAKE,
        ctx=_ctx(),
        coro_factory=_boom,
        timeout_seconds=1.0,
        event_bus=None,
    )
    assert result.action is HookAction.BLOCK
    assert result.reason is not None
    assert result.reason.startswith("hook_error:demo:")


@pytest.mark.asyncio
async def test_pipeline_timeout_blocks_and_emits_event() -> None:
    bus = RuntimeEventBus()
    events: list = []
    bus.subscribe(lambda event: events.append(event))
    pipeline = MiddlewarePipeline(
        middleware=[_SlowMiddleware()],
        hook_timeout_seconds=0.01,
        event_bus=bus,
    )
    result = await pipeline.run_before(HookPoint.BEFORE_TASK_INTAKE, _ctx())
    assert result.action is HookAction.BLOCK
    assert result.reason is not None
    assert result.reason.startswith("hook_timeout:")
    assert any(
        event.event_type is RuntimeEventType.DOMAIN_SIGNAL
        and event.event_kind == "platform.hook.hook_timeout"
        for event in events
    )


@pytest.mark.asyncio
async def test_pipeline_exception_blocks_with_hook_error() -> None:
    pipeline = MiddlewarePipeline(
        middleware=[_RaisingMiddleware()],
        hook_timeout_seconds=1.0,
        event_bus=None,
    )
    result = await pipeline.run_before(HookPoint.BEFORE_TASK_INTAKE, _ctx())
    assert result.action is HookAction.BLOCK
    assert result.reason is not None
    assert "hook_error:raising_middleware" in result.reason


@pytest.mark.asyncio
async def test_pipeline_non_allow_emits_hook_blocked_event() -> None:
    bus = RuntimeEventBus()
    events: list = []
    bus.subscribe(lambda event: events.append(event))
    pipeline = MiddlewarePipeline(
        middleware=[_BlockingMiddleware()],
        hook_timeout_seconds=1.0,
        event_bus=bus,
    )
    result = await pipeline.run_before(HookPoint.BEFORE_TASK_INTAKE, _ctx())
    assert result.action is HookAction.BLOCK
    assert any(
        event.event_type is RuntimeEventType.DOMAIN_SIGNAL
        and event.event_kind == "platform.hook.hook_blocked"
        for event in events
    )


@pytest.mark.asyncio
async def test_nexus_lifecycle_coordinator_fails_closed_on_hook_error() -> None:
    from intergrax.runtime.hooks.nexus_lifecycle_hooks import NexusLifecycleHookCoordinator
    from intergrax.runtime.task.task import Task

    pipeline = MiddlewarePipeline(middleware=[_RaisingMiddleware()], hook_timeout_seconds=1.0)
    coordinator = NexusLifecycleHookCoordinator(pipeline)
    task = Task(task_id="task-hook-err", tenant_id="t1", user_id="u1", message="hi")
    with pytest.raises(NexusLifecycleHookError, match="hook_error"):
        await coordinator.before(
            HookPoint.BEFORE_TASK_INTAKE,
            task,
            phase=ExecutionPhase.INTAKE,
        )
