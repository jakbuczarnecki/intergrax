# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""§42 decision / interrupt / retry hook helpers (Phase Q-N.5)."""

from __future__ import annotations

from typing import Optional

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.hooks.hook_context import HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline


async def run_hook_pair(
    pipeline: Optional[MiddlewarePipeline],
    before: HookPoint,
    after: HookPoint,
    ctx: HookContext,
) -> HookResult:
    if pipeline is None:
        return HookResult()
    before_result = await pipeline.run_before(before, ctx)
    if before_result.action.value != "allow":
        return before_result
    return await pipeline.run_after(after, ctx)


def hook_context_for_task(
    *,
    task_id: str,
    run_id: str,
    agent_id: Optional[str] = None,
    step_id: Optional[str] = None,
    phase: ExecutionPhase = ExecutionPhase.STEP_EXECUTION,
    runtime_state: Optional[dict] = None,
) -> HookContext:
    return HookContext(
        task_id=task_id,
        run_id=run_id,
        agent_id=agent_id,
        step_id=step_id,
        phase=phase,
        runtime_state=dict(runtime_state or {}),
    )
