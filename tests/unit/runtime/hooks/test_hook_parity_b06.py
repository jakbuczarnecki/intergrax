# © Artur Czarnecki. All rights reserved.

import pytest

from echo.echo_agent import EchoAgent
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.hooks.parity import HookCoverage, hook_coverage, list_hook_points_by_coverage
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState

pytestmark = pytest.mark.unit


@pytest.mark.gate
def test_hook_parity_documents_wired_lifecycle_points() -> None:
    wired = set(list_hook_points_by_coverage(HookCoverage.WIRED))
    assert HookPoint.BEFORE_TASK_INTAKE in wired
    assert HookPoint.AFTER_PLANNING in wired
    assert HookPoint.BEFORE_HUMAN_APPROVAL in wired
    assert HookPoint.BEFORE_STEP in wired
    assert hook_coverage(HookPoint.BEFORE_TOOL_CALL) == HookCoverage.WIRED
    assert hook_coverage(HookPoint.BEFORE_AGENT_SELECTION) == HookCoverage.WIRED


@pytest.mark.asyncio
@pytest.mark.gate
async def test_nexus_loop_lifecycle_hook_blocks_before_planning() -> None:
    observed: list[str] = []

    async def before_planning(_ctx: HookContext) -> HookResult:
        observed.append("before_planning")
        return HookResult(action=HookAction.BLOCK, reason="planning denied")

    pipeline = MiddlewarePipeline()
    pipeline.hooks.register(HookPoint.BEFORE_PLANNING, before_planning)

    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(registry, middleware=pipeline)

    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="hello",
            context=TaskContext(capability="echo.basic"),
        )
    )

    assert observed == ["before_planning"]
    assert result.state == TaskState.FAILED
    assert any("planning denied" in err for err in result.summary.validation.errors)


@pytest.mark.asyncio
@pytest.mark.gate
async def test_nexus_loop_emits_intake_and_classification_hooks() -> None:
    observed: list[HookPoint] = []

    def make_handler(point: HookPoint):
        async def _handler(_ctx: HookContext) -> HookResult:
            observed.append(point)
            return HookResult()

        return _handler

    pipeline = MiddlewarePipeline()
    for point in (
        HookPoint.BEFORE_TASK_INTAKE,
        HookPoint.AFTER_TASK_INTAKE,
        HookPoint.BEFORE_CLASSIFICATION,
        HookPoint.AFTER_CLASSIFICATION,
        HookPoint.BEFORE_PLANNING,
        HookPoint.AFTER_PLANNING,
        HookPoint.BEFORE_FINALIZATION,
        HookPoint.AFTER_FINALIZATION,
    ):
        pipeline.hooks.register(point, make_handler(point))

    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(registry, middleware=pipeline)

    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="hello",
            context=TaskContext(capability="echo.basic"),
        )
    )

    assert result.state == TaskState.COMPLETED
    assert observed[:6] == [
        HookPoint.BEFORE_TASK_INTAKE,
        HookPoint.AFTER_TASK_INTAKE,
        HookPoint.BEFORE_CLASSIFICATION,
        HookPoint.AFTER_CLASSIFICATION,
        HookPoint.BEFORE_PLANNING,
        HookPoint.AFTER_PLANNING,
    ]
    assert HookPoint.BEFORE_FINALIZATION in observed
    assert HookPoint.AFTER_FINALIZATION in observed
