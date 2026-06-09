# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.application_guardrail_middleware import LlmGuardrailMiddleware
from intergrax.applications.contracts.environment_profile import GuardrailProfile
from intergrax.integrations.providers.llm_guardrail._factory import create_guardrail_backend
from intergrax.runtime.hooks.hook_context import HookAction, HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.contracts.execution_phase import ExecutionPhase

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_guardrail_blocks_output_pattern() -> None:
    backend = create_guardrail_backend("llm_guard")
    middleware = LlmGuardrailMiddleware(
        backend,
        GuardrailProfile(enabled=True, scan_output=True),
    )
    ctx = HookContext(
        task_id="run-1",
        run_id="run-1",
        phase=ExecutionPhase.COMPLETION,
        runtime_state={
            "prompt": "summarize",
            "llm_output": "response with BLOCK_OUTPUT token",
        },
    )
    result = await middleware.after(HookPoint.AFTER_FINALIZATION, ctx)
    assert result.action == HookAction.BLOCK


@pytest.mark.asyncio
async def test_guardrail_after_llm_output_hook() -> None:
    backend = create_guardrail_backend("llm_guard")
    middleware = LlmGuardrailMiddleware(
        backend,
        GuardrailProfile(enabled=True, scan_output=True),
    )
    ctx = HookContext(
        task_id="run-2",
        run_id="run-2",
        phase=ExecutionPhase.STEP_EXECUTION,
        runtime_state={"prompt": "hi", "llm_output": "BLOCK_OUTPUT"},
    )
    result = await middleware.after(HookPoint.AFTER_LLM_OUTPUT, ctx)
    assert result.action == HookAction.BLOCK
