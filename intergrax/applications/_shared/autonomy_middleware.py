# © Artur Czarnecki. All rights reserved.

"""Autonomy-level tool gating middleware (REL-ADV.3)."""

from __future__ import annotations

from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.base import RuntimeMiddleware
from intergrax.runtime.policy.autonomy_resolver import resolve_effective_autonomy, tool_allowed_for_autonomy


class AutonomyGovernanceMiddleware(RuntimeMiddleware):
    """Enforce effective autonomy before side-effectful tool calls."""

    priority = 56
    name = "AutonomyGovernanceMiddleware"

    def __init__(
        self,
        *,
        execution_mode: ExecutionMode = ExecutionMode.BALANCED,
        default_autonomy: AutonomyLevel = AutonomyLevel.ASK,
        tenant_ceiling: AutonomyLevel | None = None,
    ) -> None:
        self._execution_mode = execution_mode
        self._default_autonomy = default_autonomy
        self._tenant_ceiling = tenant_ceiling

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        if point != HookPoint.BEFORE_TOOL_CALL:
            return HookResult()
        tool_id = str(ctx.runtime_state.get("tool_id", ""))
        if not tool_id:
            return HookResult()
        requested_raw = ctx.runtime_state.get("autonomy_level")
        requested = (
            AutonomyLevel(str(requested_raw))
            if requested_raw
            else self._default_autonomy
        )
        agent_risk = ctx.runtime_state.get("agent_risk_level")
        effective = resolve_effective_autonomy(
            requested=requested,
            execution_mode=self._execution_mode,
            agent_risk=str(agent_risk) if agent_risk is not None else None,
            tenant_ceiling=self._tenant_ceiling,
        )
        allowed, reason = tool_allowed_for_autonomy(tool_id, effective)
        if not allowed:
            return HookResult(action=HookAction.BLOCK, reason=reason)
        return HookResult()

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult()
