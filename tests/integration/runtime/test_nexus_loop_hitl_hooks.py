# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.human.hitl_hooks import (
    HumanApprovalHookCoordinator,
    HumanApprovalHookError,
    human_approval_hook_context,
)
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from testing_support.nexus_hitl_test_agent import NexusBasicHitlTestAgent

_HitlAgent = NexusBasicHitlTestAgent


@pytest.mark.unit
@pytest.mark.gate
def test_human_approval_hook_context_includes_phase():
    task = Task(tenant_id="t1", user_id="u1", message="x", agent_id="hitl")
    ctx = human_approval_hook_context(task, verdict="approve")
    assert ctx.phase == ExecutionPhase.HUMAN_APPROVAL
    assert ctx.runtime_state["human_verdict"] == "approve"


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_human_approval_hook_coordinator_blocks_before_pause():
    pipeline = MiddlewarePipeline()

    async def block_handler(_ctx: HookContext) -> HookResult:
        return HookResult(action=HookAction.BLOCK, reason="policy denied")

    pipeline.hooks.register(HookPoint.BEFORE_HUMAN_APPROVAL, block_handler)
    coordinator = HumanApprovalHookCoordinator(pipeline)
    task = Task(tenant_id="t1", user_id="u1", message="x")

    with pytest.raises(HumanApprovalHookError, match="policy denied"):
        await coordinator.before_pause(task)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_emits_hitl_hooks_on_pause_and_resume():
    observed: list[tuple[str, str]] = []

    async def before_handler(ctx: HookContext) -> HookResult:
        observed.append(("before", ctx.task_id))
        return HookResult()

    async def after_handler(ctx: HookContext) -> HookResult:
        observed.append(("after", str(ctx.runtime_state.get("human_verdict"))))
        return HookResult()

    pipeline = MiddlewarePipeline()
    pipeline.hooks.register(HookPoint.BEFORE_HUMAN_APPROVAL, before_handler)
    pipeline.hooks.register(HookPoint.AFTER_HUMAN_APPROVAL, after_handler)

    registry = AgentRegistry()
    registry.register(_HitlAgent())
    loop = NexusLoop(registry, middleware=pipeline)

    paused = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
        )
    )

    assert paused.state == TaskState.WAITING_FOR_HUMAN
    assert ("before", paused.task_id) in observed
    assert not any(item[0] == "after" for item in observed)

    completed = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
            task_id=paused.task_id,
            metadata={"human_approved": True},
        )
    )

    assert completed.state == TaskState.COMPLETED
    assert ("after", "approve") in observed


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_resume_not_blocked_by_allow_hooks():
    pipeline = MiddlewarePipeline()
    pipeline.hooks.register(
        HookPoint.BEFORE_HUMAN_APPROVAL,
        lambda _ctx: HookResult(),
    )
    pipeline.hooks.register(
        HookPoint.AFTER_HUMAN_APPROVAL,
        lambda _ctx: HookResult(),
    )

    registry = AgentRegistry()
    registry.register(_HitlAgent())
    loop = NexusLoop(registry, middleware=pipeline)

    paused = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
        )
    )
    assert paused.state == TaskState.WAITING_FOR_HUMAN

    completed = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
            task_id=paused.task_id,
            metadata={"human_approved": True},
        )
    )
    assert completed.state == TaskState.COMPLETED
