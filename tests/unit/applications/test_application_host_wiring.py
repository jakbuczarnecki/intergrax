# © Artur Czarnecki. All rights reserved.

"""APP-CON-1 — ApplicationHost mounted on Nexus middleware pipeline."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.harness.application_host import ApplicationHost
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _BlockSelectionHost:
    def on_hook(self, point: HookPoint, context: HookContext) -> HookResult | None:
        if point == HookPoint.BEFORE_AGENT_SELECTION:
            return HookResult(action=HookAction.BLOCK, reason="test_block")
        return None


def test_build_harness_host_runtime_mounts_application_host() -> None:
    manifest = ApplicationManifest.lab(
        app_id="host_wiring_test",
        name="Host Wiring Test",
        route_prefix="/v1/host_wiring_test",
        env_prefix="HOST_WIRING_TEST_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="host_wiring_test.lab")
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        use_in_memory_trace=True,
        application_host=_BlockSelectionHost(),
    )
    pipeline = runtime.nexus_loop.middleware
    assert isinstance(pipeline, MiddlewarePipeline)
    names = [mw.name for mw in pipeline._middleware]  # noqa: SLF001
    assert "application_host" in names


@pytest.mark.asyncio
async def test_application_host_blocks_agent_selection() -> None:
    manifest = ApplicationManifest.lab(
        app_id="host_block_test",
        name="Host Block Test",
        route_prefix="/v1/host_block_test",
        env_prefix="HOST_BLOCK_TEST_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="host_block_test.lab")
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        use_in_memory_trace=True,
        application_host=_BlockSelectionHost(),
    )
    task = Task(
        task_id="task-host-block-1",
        tenant_id="tenant-test",
        user_id="user-test",
        message="hello",
        context=TaskContext(capability="echo.basic"),
    )
    result = await runtime.nexus_loop.handle_task(task)
    assert result.state.value == "failed"
