# © Artur Czarnecki. All rights reserved.

"""Security plane domain signals on the observability spine (Phase SEC-EVOL-3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.events.spine_consolidation import build_platform_signal_event
from intergrax.runtime.hooks.hook_context import HookContext
from intergrax.runtime.hooks.hook_point import HookPoint

if TYPE_CHECKING:
    from intergrax.runtime.events.event_bus import RuntimeEventBus

KIND_DEFENSE_BLOCKED = "platform.security.defense_blocked"
KIND_ENCRYPTION_DENIED = "platform.security.encryption_denied"


async def emit_defense_blocked(
    event_bus: RuntimeEventBus | None,
    *,
    ctx: HookContext,
    point: HookPoint,
    plugin_id: str,
    reason: str,
) -> None:
    if event_bus is None:
        return
    await event_bus.publish(
        build_platform_signal_event(
            kind=KIND_DEFENSE_BLOCKED,
            tenant_id=str(ctx.runtime_state.get("tenant_id", "")) or None,
            task_id=ctx.task_id,
            run_id=ctx.run_id,
            node_id=ctx.node_id,
            agent_id=ctx.agent_id,
            step_id=ctx.step_id,
            phase=ctx.phase,
            severity=EventSeverity.WARNING,
            correlation_id=ctx.task_id,
            payload={
                "plugin_id": plugin_id,
                "hook_point": point.value,
                "reason": reason,
            },
        )
    )


async def emit_encryption_denied(
    event_bus: RuntimeEventBus | None,
    *,
    ctx: HookContext,
    point: HookPoint,
    reason: str,
    classification: str | None = None,
) -> None:
    if event_bus is None:
        return
    await event_bus.publish(
        build_platform_signal_event(
            kind=KIND_ENCRYPTION_DENIED,
            tenant_id=str(ctx.runtime_state.get("tenant_id", "")) or None,
            task_id=ctx.task_id,
            run_id=ctx.run_id,
            node_id=ctx.node_id,
            agent_id=ctx.agent_id,
            step_id=ctx.step_id,
            phase=ctx.phase,
            severity=EventSeverity.WARNING,
            correlation_id=ctx.task_id,
            payload={
                "hook_point": point.value,
                "reason": reason,
                "classification": classification,
            },
        )
    )
