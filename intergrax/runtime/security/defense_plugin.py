# © Artur Czarnecki. All rights reserved.

"""Security defense plugin contract (Phase SEC-EXT-1)."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.base import RuntimeMiddleware
from intergrax.runtime.security.security_events import emit_defense_blocked

if TYPE_CHECKING:
    from intergrax.runtime.events.event_bus import RuntimeEventBus

DEFAULT_DEFENSE_INSPECTION_TIMEOUT_MS = 100


class SecurityFailMode(str, Enum):
    """How a defense plugin should fail when inspection blocks."""

    FAIL_CLOSED = "fail_closed"
    FAIL_OPEN = "fail_open"


class SecurityInspectionResult(BaseModel):
    """Outcome of a defense plugin inspection at a UAEP hook."""

    model_config = {"extra": "forbid"}

    allowed: bool = True
    reasons: list[str] = Field(default_factory=list)
    plugin_id: str = ""
    hook_point: str = ""


@runtime_checkable
class SecurityDefensePlugin(Protocol):
    """Author-facing contract for ``intergrax.security_defenses`` entry points."""

    plugin_id: str
    version: str
    hook_points: frozenset[HookPoint]
    priority: int
    fail_mode: SecurityFailMode

    def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult: ...


class PluginSecurityDefenseMiddleware(RuntimeMiddleware):
    """Wrap a :class:`SecurityDefensePlugin` as Tier-1 runtime middleware."""

    def __init__(
        self,
        plugin: SecurityDefensePlugin,
        *,
        event_bus: RuntimeEventBus | None = None,
        inspection_timeout_ms: int = DEFAULT_DEFENSE_INSPECTION_TIMEOUT_MS,
    ) -> None:
        self._plugin = plugin
        self._event_bus = event_bus
        self._inspection_timeout_ms = inspection_timeout_ms
        self.priority = plugin.priority
        self.name = f"SecurityDefense:{plugin.plugin_id}"

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        if point not in self._plugin.hook_points:
            return HookResult()
        timeout_seconds = max(self._inspection_timeout_ms, 1) / 1000.0
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._plugin.inspect, point, ctx),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            reason = f"defense plugin inspection timeout: {self._plugin.plugin_id}"
            await emit_defense_blocked(
                self._event_bus,
                ctx=ctx,
                point=point,
                plugin_id=self._plugin.plugin_id,
                reason=reason,
            )
            return HookResult(action=HookAction.BLOCK, reason=reason)
        if result.allowed:
            return HookResult()
        reason = "; ".join(result.reasons) or f"blocked by {self._plugin.plugin_id}"
        await emit_defense_blocked(
            self._event_bus,
            ctx=ctx,
            point=point,
            plugin_id=self._plugin.plugin_id,
            reason=reason,
        )
        if self._plugin.fail_mode == SecurityFailMode.FAIL_OPEN:
            return HookResult(action=HookAction.MODIFY, reason=reason)
        return HookResult(action=HookAction.BLOCK, reason=reason)

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult()
