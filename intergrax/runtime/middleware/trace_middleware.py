# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Trace emission middleware stub (architecture §42.24)."""

from __future__ import annotations

from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_context import HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.base import RuntimeMiddleware


def _tenant_id_from_hook(ctx: HookContext) -> str:
    raw = ctx.runtime_state.get("tenant_id")
    if raw:
        return str(raw)
    return "default"


class TraceEmittingMiddleware(RuntimeMiddleware):
    """Publishes ``RuntimeEvent`` on step start (UAEP-MAINT-02).

    ``STEP_COMPLETED`` is emitted only by ``HarnessKernel`` to avoid duplicate
    journal entries when UAEP runs through the step kernel bridge.
    """

    priority = 10
    name = "TraceEmittingMiddleware"

    def __init__(self, bus: RuntimeEventBus) -> None:
        self._bus = bus

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        if point == HookPoint.BEFORE_STEP and ctx.step_id:
            event = RuntimeEvent(
                task_id=ctx.task_id,
                run_id=ctx.run_id,
                node_id=ctx.node_id,
                agent_id=ctx.agent_id,
                step_id=ctx.step_id,
                tenant_id=_tenant_id_from_hook(ctx),
                event_type=RuntimeEventType.STEP_STARTED,
                phase=ctx.phase,
                correlation_id=ctx.task_id,
            )
            await self._bus.publish(event)
        return HookResult()

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        _ = point, ctx
        return HookResult()
