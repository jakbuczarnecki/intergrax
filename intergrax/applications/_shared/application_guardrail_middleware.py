# © Artur Czarnecki. All rights reserved.

"""Vendor LLM guardrail middleware (M-P12-WIRE.1)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import GuardrailProfile
from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import (
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.contracts.host_orchestration_wiring_capabilities import (
    HostOrchestrationRuntimeEventPort,
)
from intergrax.contracts.middleware_hook_semantics import (
    LlmInferenceHookPayload,
    ToolCallHookPayload,
)
from intergrax.integrations.contracts.llm_guardrail import GuardrailContext, LlmGuardrailBackend
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
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
        *,
        event_bus: HostOrchestrationRuntimeEventPort | None = None,
    ) -> None:
        self._backend = backend
        self._profile = profile
        self._event_bus = event_bus

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        if not self._profile.enabled:
            return HookResult()
        guard_ctx = _guardrail_context(ctx, point)
        if point in {HookPoint.BEFORE_CONTEXT_BUILD, HookPoint.BEFORE_LLM_INFERENCE} and self._profile.scan_input:
            llm_payload = ctx.payload
            if not isinstance(llm_payload, LlmInferenceHookPayload):
                return HookResult()
            prompt = llm_payload.prompt or ""
            if not prompt:
                return HookResult()
            result = self._backend.scan_input(prompt, context=guard_ctx)
            if not result.allowed:
                reason = result.detail or f"guardrail input blocked ({self._backend.slug})"
                await self._emit_blocked(
                    ctx,
                    point=point,
                    scan_kind="input",
                    reason=reason,
                )
                return HookResult(
                    action=HookAction.BLOCK,
                    reason=reason,
                    modified_payload={"guardrail": result.audit_payload},
                )
            if result.sanitized_text and result.sanitized_text != prompt:
                return HookResult(
                    modified_payload={"prompt": result.sanitized_text},
                )
        if point == HookPoint.BEFORE_TOOL_CALL and self._profile.scan_tool_calls:
            tool_payload = ctx.payload
            if not isinstance(tool_payload, ToolCallHookPayload):
                return HookResult()
            tool_id = tool_payload.tool_id
            if not tool_id:
                return HookResult()
            arguments = dict(tool_payload.arguments)
            result = self._backend.scan_tool_call(tool_id, arguments, context=guard_ctx)
            if not result.allowed:
                reason = result.detail or f"guardrail tool blocked ({self._backend.slug})"
                await self._emit_blocked(
                    ctx,
                    point=point,
                    scan_kind="tool",
                    reason=reason,
                )
                return HookResult(
                    action=HookAction.BLOCK,
                    reason=reason,
                )
        return HookResult()

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        if not self._profile.enabled or not self._profile.scan_output:
            return HookResult()
        if point not in {HookPoint.AFTER_LLM_OUTPUT, HookPoint.AFTER_FINALIZATION}:
            return HookResult()
        llm_payload = ctx.payload
        if not isinstance(llm_payload, LlmInferenceHookPayload):
            return HookResult()
        output = llm_payload.llm_output or ""
        if not output:
            return HookResult()
        prompt = llm_payload.prompt or ""
        result = self._backend.scan_output(
            output,
            context=_guardrail_context(ctx, point),
            prompt=prompt or None,
        )
        if not result.allowed:
            reason = result.detail or f"guardrail output blocked ({self._backend.slug})"
            await self._emit_blocked(
                ctx,
                point=point,
                scan_kind="output",
                reason=reason,
            )
            return HookResult(
                action=HookAction.BLOCK,
                reason=reason,
                modified_payload={
                    "guardrail_scan": {
                        **result.audit_payload,
                        "allowed": False,
                        "categories": list(result.categories),
                        "detail": result.detail,
                    },
                },
            )
        return HookResult(
            modified_payload={
                "guardrail_scan": {
                    **result.audit_payload,
                    "allowed": True,
                    "categories": list(result.categories),
                    "detail": result.detail,
                },
            },
        )

    async def _emit_blocked(
        self,
        ctx: HookContext,
        *,
        point: HookPoint,
        scan_kind: str,
        reason: str,
    ) -> None:
        if self._event_bus is None:
            return
        active_run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        if ctx.run_id != active_run_id:
            raise RuntimeError("guardrail run_id conflicts with active execution identity")
        await self._event_bus.publish(
            RuntimeEvent(
                tenant_id=ctx.subject.tenant_id,
                task_id=ctx.task_id,
                run_id=ctx.run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                node_id=ctx.node_id,
                agent_id=ctx.agent_id,
                step_id=ctx.step_id,
                event_type=RuntimeEventType.GUARDRAIL_BLOCKED,
                phase=ctx.phase,
                severity=EventSeverity.WARNING,
                correlation_id=ctx.task_id,
                payload={
                    "scan_kind": scan_kind,
                    "hook": point.value,
                    "backend_slug": self._backend.slug,
                    "reason": reason,
                },
            )
        )


def _guardrail_context(ctx: HookContext, point: HookPoint) -> GuardrailContext:
    return GuardrailContext(
        tenant_id=ctx.subject.tenant_id or "",
        run_id=ctx.run_id,
        agent_id=ctx.agent_id or "",
        step_id=ctx.step_id or ctx.node_id or "",
        hook=point.value,
    )


def _stringify_argument_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}
