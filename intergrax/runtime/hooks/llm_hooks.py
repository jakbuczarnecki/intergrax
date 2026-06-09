# © Artur Czarnecki. All rights reserved.

"""LLM inference hook helpers (§42.20, M.12 follow-up)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint

if TYPE_CHECKING:
    from intergrax.runtime.middleware.pipeline import MiddlewarePipeline


class LlmGuardrailBlockedError(RuntimeError):
    """Raised when guardrail middleware blocks LLM input or output."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def llm_hook_context(
    *,
    run_id: str,
    prompt: str,
    agent_id: str | None = None,
    step_id: str | None = None,
    tenant_id: str = "",
) -> HookContext:
    return HookContext(
        task_id=run_id,
        run_id=run_id,
        agent_id=agent_id,
        step_id=step_id,
        phase=ExecutionPhase.STEP_EXECUTION,
        runtime_state={
            "prompt": prompt,
            "tenant_id": tenant_id,
        },
    )


async def run_llm_generation_hooks(
    middleware: MiddlewarePipeline | None,
    *,
    ctx: HookContext,
    prompt: str,
    generate: Callable[[], str],
) -> str:
    """Run BEFORE_LLM_INFERENCE / AFTER_LLM_OUTPUT around synchronous LLM generation."""
    if middleware is None:
        return generate()

    inference_ctx = ctx.model_copy(
        update={
            "runtime_state": {
                **ctx.runtime_state,
                "prompt": prompt,
            },
        },
    )
    before = await middleware.run_before(HookPoint.BEFORE_LLM_INFERENCE, inference_ctx)
    if before.action == HookAction.BLOCK:
        raise LlmGuardrailBlockedError(before.reason or "llm_input_blocked_by_guardrail")

    text = generate()

    output_ctx = inference_ctx.model_copy(
        update={
            "runtime_state": {
                **inference_ctx.runtime_state,
                "llm_output": text,
                "output": text,
            },
        },
    )
    after = await middleware.run_after(HookPoint.AFTER_LLM_OUTPUT, output_ctx)
    if after.action == HookAction.BLOCK:
        raise LlmGuardrailBlockedError(after.reason or "llm_output_blocked_by_guardrail")
    return text


def guard_hook_result(result: HookResult, *, point: HookPoint) -> None:
    if result.action != HookAction.ALLOW:
        raise LlmGuardrailBlockedError(
            result.reason or f"guardrail blocked at {point.value}: {result.action.value}",
        )
