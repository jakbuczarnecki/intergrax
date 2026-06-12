# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hook registry (architecture §42.3, §42.22)."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, DefaultDict, Dict, List, Optional, Union
from uuid import uuid4

from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint

logger = logging.getLogger(__name__)

HookHandler = Callable[[HookContext], Union[HookResult, Awaitable[HookResult]]]


class HookRegistration:
    __slots__ = ("hook_id", "point", "priority", "handler", "name")

    def __init__(
        self,
        hook_id: str,
        point: HookPoint,
        priority: int,
        handler: HookHandler,
        name: str,
    ) -> None:
        self.hook_id = hook_id
        self.point = point
        self.priority = priority
        self.handler = handler
        self.name = name


class HookRegistry:
    """Ordered hook handlers per ``HookPoint``."""

    def __init__(self) -> None:
        self._hooks: Dict[HookPoint, List[HookRegistration]] = {}

    def register(
        self,
        point: HookPoint,
        handler: HookHandler,
        *,
        priority: int = 100,
        name: Optional[str] = None,
        hook_id: Optional[str] = None,
    ) -> str:
        hid = hook_id or f"hook_{uuid4().hex[:8]}"
        reg = HookRegistration(
            hook_id=hid,
            point=point,
            priority=priority,
            handler=handler,
            name=name or handler.__name__,
        )
        self._hooks.setdefault(point, []).append(reg)
        self._hooks[point].sort(key=lambda r: r.priority)
        return hid

    def unregister(self, hook_id: str) -> None:
        for point in list(self._hooks.keys()):
            self._hooks[point] = [r for r in self._hooks[point] if r.hook_id != hook_id]

    async def run(self, point: HookPoint, ctx: HookContext) -> HookResult:
        aggregate = HookResult(action=HookAction.ALLOW)
        for reg in self._hooks.get(point, []):
            try:
                raw = reg.handler(ctx)
                result = await raw if hasattr(raw, "__await__") else raw
            except Exception:
                logger.exception("Hook %s failed at %s", reg.name, point.value)
                return HookResult(action=HookAction.BLOCK, reason=f"hook_error:{reg.name}")
            if result.action == HookAction.BLOCK:
                return result
            if result.action == HookAction.ESCALATE:
                return result
            if result.action == HookAction.MODIFY and result.modified_payload:
                ctx.runtime_state.update(result.modified_payload)
                aggregate = result
        return aggregate

    def list_hooks(self, point: Optional[HookPoint] = None) -> List[HookRegistration]:
        if point is not None:
            return list(self._hooks.get(point, []))
        out: List[HookRegistration] = []
        for regs in self._hooks.values():
            out.extend(regs)
        return out
