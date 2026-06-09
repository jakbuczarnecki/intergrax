# © Artur Czarnecki. All rights reserved.

"""Vendor LLM guardrail middleware (M-P12-WIRE.1)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import GuardrailProfile
from intergrax.integrations.contracts.llm_guardrail import GuardrailContext, LlmGuardrailBackend
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.base import RuntimeMiddleware


class LlmGuardrailMiddleware(RuntimeMiddleware):
    """Scan prompts, outputs, and optional tool args via catalog guardrail backend."""

    priority = 52
    name = "LlmGuardrailMiddleware"

    def __init__(
        self,
        backend: LlmGuardrailBackend,
        profile: GuardrailProfile,
    ) -> None:
        self._backend = backend
        self._profile = profile

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        if not self._profile.enabled:
            return HookResult()
        guard_ctx = _guardrail_context(ctx, point)
        if point in {HookPoint.BEFORE_CONTEXT_BUILD, HookPoint.BEFORE_LLM_INFERENCE} and self._profile.scan_input:
            prompt = str(ctx.runtime_state.get("prompt", ""))
            if not prompt:
                return HookResult()
            result = self._backend.scan_input(prompt, context=guard_ctx)
            if not result.allowed:
                return HookResult(
                    action=HookAction.BLOCK,
                    reason=result.detail or f"guardrail input blocked ({self._backend.slug})",
                    modified_payload={"guardrail": result.audit_payload},
                )
            if result.sanitized_text and result.sanitized_text != prompt:
                return HookResult(
                    modified_payload={"prompt": result.sanitized_text},
                )
        if point == HookPoint.BEFORE_TOOL_CALL and self._profile.scan_tool_calls:
            tool_id = str(ctx.runtime_state.get("tool_id", ctx.runtime_state.get("tool_name", "")))
            if not tool_id:
                return HookResult()
            arguments = _stringify_argument_map(ctx.runtime_state.get("arguments"))
            result = self._backend.scan_tool_call(tool_id, arguments, context=guard_ctx)
            if not result.allowed:
                return HookResult(
                    action=HookAction.BLOCK,
                    reason=result.detail or f"guardrail tool blocked ({self._backend.slug})",
                )
        return HookResult()

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        if not self._profile.enabled or not self._profile.scan_output:
            return HookResult()
        if point not in {HookPoint.AFTER_LLM_OUTPUT, HookPoint.AFTER_FINALIZATION}:
            return HookResult()
        output = str(ctx.runtime_state.get("llm_output", ctx.runtime_state.get("output", "")))
        if not output:
            return HookResult()
        prompt = str(ctx.runtime_state.get("prompt", ""))
        result = self._backend.scan_output(
            output,
            context=_guardrail_context(ctx, point),
            prompt=prompt or None,
        )
        if not result.allowed:
            return HookResult(
                action=HookAction.BLOCK,
                reason=result.detail or f"guardrail output blocked ({self._backend.slug})",
            )
        return HookResult()


def _guardrail_context(ctx: HookContext, point: HookPoint) -> GuardrailContext:
    return GuardrailContext(
        tenant_id=str(ctx.runtime_state.get("tenant_id", "")),
        run_id=ctx.run_id,
        agent_id=ctx.agent_id or "",
        step_id=ctx.step_id or ctx.node_id or "",
        hook=point.value,
    )


def _stringify_argument_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}
