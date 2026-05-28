# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""HITL middleware hooks for NexusLoop (Phase G.3, §42.10)."""

from __future__ import annotations

from typing import Optional

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.task.task import Task


class HumanApprovalHookError(RuntimeError):
    """Raised when a HITL middleware hook blocks execution."""


def human_approval_hook_context(
    task: Task,
    *,
    agent_id: Optional[str] = None,
    execution: Optional[AgentExecutionResult] = None,
    verdict: Optional[str] = None,
) -> HookContext:
    gov = task.runtime.governance
    runtime_state = {
        "task_state": task.state.value,
        "governance_paused": gov.paused,
    }
    if gov.human_request is not None:
        runtime_state["human_request"] = gov.human_request.model_dump()
    if execution is not None:
        runtime_state["execution_status"] = execution.status.value
        if execution.human_request is not None:
            runtime_state["human_request"] = execution.human_request.model_dump()
    if verdict is not None:
        runtime_state["human_verdict"] = verdict

    return HookContext(
        task_id=task.task_id,
        run_id=task.task_id,
        agent_id=agent_id or task.agent_id or (execution.agent_id if execution else None),
        phase=ExecutionPhase.HUMAN_APPROVAL,
        runtime_state=runtime_state,
    )


class HumanApprovalHookCoordinator:
    """Runs BEFORE/AFTER_HUMAN_APPROVAL through the shared middleware pipeline."""

    def __init__(self, pipeline: MiddlewarePipeline) -> None:
        self._pipeline = pipeline

    @property
    def pipeline(self) -> MiddlewarePipeline:
        return self._pipeline

    async def before_pause(
        self,
        task: Task,
        *,
        agent_id: Optional[str] = None,
        execution: Optional[AgentExecutionResult] = None,
    ) -> None:
        ctx = human_approval_hook_context(task, agent_id=agent_id, execution=execution)
        await self._guard(
            await self._pipeline.run_before(HookPoint.BEFORE_HUMAN_APPROVAL, ctx),
            HookPoint.BEFORE_HUMAN_APPROVAL,
        )

    async def after_response(
        self,
        task: Task,
        *,
        verdict: str,
        agent_id: Optional[str] = None,
    ) -> None:
        ctx = human_approval_hook_context(task, agent_id=agent_id, verdict=verdict)
        await self._guard(
            await self._pipeline.run_after(HookPoint.AFTER_HUMAN_APPROVAL, ctx),
            HookPoint.AFTER_HUMAN_APPROVAL,
        )

    @staticmethod
    async def _guard(result: HookResult, point: HookPoint) -> None:
        if result.action != HookAction.ALLOW:
            raise HumanApprovalHookError(
                result.reason or f"hook blocked at {point.value}: {result.action.value}"
            )
