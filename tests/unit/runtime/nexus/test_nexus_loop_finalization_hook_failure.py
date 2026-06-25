# © Artur Czarnecki. All rights reserved.

"""LKW-H0.3 — post-finalization lifecycle hook failures must be visible."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _hook_failure_events(loop: NexusLoop, *, point: str) -> list:
    return [
        event
        for event in loop.event_bus.history
        if event.event_type is RuntimeEventType.DOMAIN_SIGNAL
        and event.event_kind
        in {
            "platform.hook.hook_blocked",
            "platform.hook.hook_error",
            "platform.hook.hook_timeout",
        }
        and event.payload.get("point") == point
        and event.task_id
    ]


@pytest.mark.asyncio
async def test_after_finalization_hook_failure_emits_runtime_event_and_keeps_success() -> None:
    async def after_finalization(_ctx: HookContext) -> HookResult:
        return HookResult(action=HookAction.BLOCK, reason="cleanup hook failed")

    pipeline = MiddlewarePipeline()
    pipeline.hooks.register(HookPoint.AFTER_FINALIZATION, after_finalization)

    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(registry, middleware=pipeline)

    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="hello",
            context=TaskContext(capability="echo.basic"),
        )
    )

    assert result.state is TaskState.COMPLETED
    events = _hook_failure_events(loop, point=HookPoint.AFTER_FINALIZATION.value)
    assert len(events) >= 1
    event = events[-1]
    assert event.task_id == result.task_id
    assert event.run_id == result.task_id
    assert event.payload.get("hook_name") == "lifecycle_hook"
    assert event.payload.get("error_type") == "NexusLifecycleHookError"
    assert "cleanup hook failed" in str(event.payload.get("reason", ""))
    assert event.payload.get("non_critical") is True


@pytest.mark.asyncio
async def test_before_finalization_hook_failure_emits_runtime_event_and_fails_task() -> None:
    async def before_finalization(_ctx: HookContext) -> HookResult:
        return HookResult(action=HookAction.BLOCK, reason="finalization denied")

    pipeline = MiddlewarePipeline()
    pipeline.hooks.register(HookPoint.BEFORE_FINALIZATION, before_finalization)

    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(registry, middleware=pipeline)

    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="hello",
            context=TaskContext(capability="echo.basic"),
        )
    )

    assert result.state is TaskState.COMPLETED
    assert result.summary.validation.valid is False
    assert any("finalization denied" in err for err in result.summary.validation.errors)
    events = _hook_failure_events(loop, point=HookPoint.BEFORE_FINALIZATION.value)
    assert len(events) >= 1
    event = events[-1]
    assert event.task_id == result.task_id
    assert "finalization denied" in str(event.payload.get("reason", ""))
    assert event.payload.get("non_critical") is False
