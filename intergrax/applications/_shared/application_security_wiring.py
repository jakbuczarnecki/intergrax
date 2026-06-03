# © Artur Czarnecki. All rights reserved.

"""Register per-application V-SEC hooks into Nexus middleware (Phase H-APP.2.7)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationSecurityProfile
from intergrax.runtime.architecture.prompt_security import (
    PromptDefenseProfile,
    PromptInjectionRule,
    PromptRiskLevel,
    inspect_prompt_for_injection,
)
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.base import RuntimeMiddleware
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def default_prompt_defense_profile() -> PromptDefenseProfile:
    return PromptDefenseProfile(
        profile_id="harness.default",
        version="1",
        rules=[
            PromptInjectionRule(
                rule_id="ignore_instructions",
                pattern="ignore previous instructions",
                risk_level=PromptRiskLevel.HIGH,
                block=True,
            ),
        ],
    )


class PromptDefenseMiddleware(RuntimeMiddleware):
    """Block prompts matching configured injection patterns."""

    priority = 50
    name = "PromptDefenseMiddleware"

    def __init__(self, profile: PromptDefenseProfile) -> None:
        self._profile = profile

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        if point != HookPoint.BEFORE_CONTEXT_BUILD:
            return HookResult()
        prompt = str(ctx.runtime_state.get("prompt", ""))
        if not prompt:
            return HookResult()
        result = inspect_prompt_for_injection(prompt=prompt, profile=self._profile)
        if result.blocked:
            return HookResult(
                action=HookAction.BLOCK,
                reason=f"Prompt blocked: {', '.join(result.reasons)}",
            )
        return HookResult()

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult()


def register_application_security_hooks(
    nexus: NexusLoop,
    profile: ApplicationSecurityProfile,
) -> None:
    """Attach security middleware when V-SEC toggles are enabled."""
    if not profile.prompt_defense_enabled:
        return
    defense = default_prompt_defense_profile()
    middleware = PromptDefenseMiddleware(defense)
    pipeline = nexus._middleware  # noqa: SLF001 — Tier-3 composition hook
    if isinstance(pipeline, MiddlewarePipeline):
        pipeline._middleware = sorted(  # noqa: SLF001
            [middleware, *pipeline._middleware],
            key=lambda item: item.priority,
        )
