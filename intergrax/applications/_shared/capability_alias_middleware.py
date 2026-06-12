# © Artur Czarnecki. All rights reserved.

"""Redirect deprecated capability tokens on task intake (APP-EVOL-3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.applications._shared.capability_alias_wiring import (
    build_capability_alias_registry,
    capability_alias_redirect_payload,
    resolve_capability_alias,
    strict_mode_for_environment,
)
from intergrax.applications.contracts.capability_alias import CAPABILITY_ALIAS_REDIRECT_KEY
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.base import RuntimeMiddleware
from intergrax.utils.time_provider import SystemTimeProvider

if TYPE_CHECKING:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


class CapabilityAliasMiddleware(RuntimeMiddleware):
    """Resolve legacy capability aliases before intake snapshot capture (priority 34)."""

    priority = 34
    name = "capability_alias"

    def __init__(self, *, environment: ApplicationEnvironmentProfile) -> None:
        self._environment = environment
        self._registry = build_capability_alias_registry(environment.capability_governance_profile)

    async def before(self, point: HookPoint, context: HookContext) -> HookResult:
        if point != HookPoint.BEFORE_TASK_INTAKE:
            return HookResult()
        if not self._registry.aliases:
            return HookResult()

        capability = context.runtime_state.get("capability")
        if not isinstance(capability, str) or not capability.strip():
            return HookResult()

        resolution = resolve_capability_alias(
            capability,
            self._registry,
            now=SystemTimeProvider.utc_now(),
            strict=strict_mode_for_environment(self._environment.execution_mode),
        )
        if resolution.blocked:
            return HookResult(action=HookAction.BLOCK, reason=resolution.reason)

        context.runtime_state["capability"] = resolution.resolved
        if resolution.redirected:
            context.runtime_state[CAPABILITY_ALIAS_REDIRECT_KEY] = capability_alias_redirect_payload(
                resolution,
            )
        return HookResult()

    async def after(self, point: HookPoint, context: HookContext) -> HookResult:
        return HookResult()
