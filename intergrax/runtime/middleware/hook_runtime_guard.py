# © Artur Czarnecki. All rights reserved.

"""Hook timeout, error→BLOCK, and audit events (architecture §32.6.5 · APP-CON-5)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.events.spine_consolidation import build_platform_signal_event

if TYPE_CHECKING:
    from intergrax.runtime.events.event_bus import RuntimeEventBus


async def invoke_guarded_hook(
    *,
    hook_name: str,
    point: HookPoint,
    ctx: HookContext,
    coro_factory: Callable[[], Awaitable[HookResult]],
    timeout_seconds: float | None,
    event_bus: RuntimeEventBus | None,
) -> HookResult:
    """Run a middleware/registry hook with wall-time cap and fail-closed error handling."""
    try:
        if timeout_seconds is not None:
            result = await asyncio.wait_for(coro_factory(), timeout=timeout_seconds)
        else:
            result = await coro_factory()
    except TimeoutError:
        result = HookResult(
            action=HookAction.BLOCK,
            reason=f"hook_timeout:{hook_name}",
        )
        await _emit_hook_violation(
            event_bus,
            point=point,
            ctx=ctx,
            hook_name=hook_name,
            result=result,
            violation_kind="timeout",
        )
        return result
    except Exception as exc:
        result = HookResult(
            action=HookAction.BLOCK,
            reason=f"hook_error:{hook_name}:{exc}",
        )
        await _emit_hook_violation(
            event_bus,
            point=point,
            ctx=ctx,
            hook_name=hook_name,
            result=result,
            violation_kind="error",
        )
        return result

    if result.action is not HookAction.ALLOW:
        await _emit_hook_violation(
            event_bus,
            point=point,
            ctx=ctx,
            hook_name=hook_name,
            result=result,
            violation_kind=result.action.value,
        )
    return result


async def _emit_hook_violation(
    event_bus: RuntimeEventBus | None,
    *,
    point: HookPoint,
    ctx: HookContext,
    hook_name: str,
    result: HookResult,
    violation_kind: str,
) -> None:
    if event_bus is None:
        return
    kind = {
        "timeout": "platform.hook.hook_timeout",
        "error": "platform.hook.hook_error",
    }.get(violation_kind, "platform.hook.hook_blocked")
    severity = (
        EventSeverity.ERROR
        if violation_kind in {"timeout", "error"}
        else EventSeverity.WARNING
    )
    await event_bus.publish(
        build_platform_signal_event(
            kind=kind,
            task_id=ctx.task_id,
            run_id=ctx.run_id,
            node_id=ctx.node_id,
            agent_id=ctx.agent_id,
            step_id=ctx.step_id,
            phase=ctx.phase,
            severity=severity,
            correlation_id=ctx.task_id,
            payload={
                "hook_name": hook_name,
                "point": point.value,
                "action": result.action.value,
                "reason": result.reason,
                "violation_kind": violation_kind,
            },
        )
    )
