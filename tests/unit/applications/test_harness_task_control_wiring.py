# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.applications._shared.task_control_wiring import (
    build_reliability_task_enricher,
    build_task_runner_with_enricher,
)
from poc_template_application.host.factory import create_poc_template_application
from poc_template_application.host.settings import PocTemplateApplicationSettings

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_build_reliability_task_enricher_applies_defaults() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.reliability_profile.default_autonomy_level = AutonomyLevel.ASK
    enricher = build_reliability_task_enricher(env)
    task = enricher(
        Task(
            task_id="t",
            tenant_id="t",
            user_id="u",
            message="m",
            context=TaskContext(capability="echo.basic"),
        )
    )
    assert task.options.governance.autonomy_level is AutonomyLevel.ASK


def test_poc_template_mounts_harness_task_routes_by_default() -> None:
    settings = PocTemplateApplicationSettings(
        include_task_control=True,
        include_mcp=False,
        include_scheduler=False,
        include_interaction_routes=False,
    )
    app = create_poc_template_application(settings=settings)
    paths = {route.path for route in app.routes}
    assert "/v1/tasks/run-async" in paths
    assert "/v1/tasks/{task_id}/cancel" in paths


@pytest.mark.asyncio
async def test_unified_task_runner_applies_enricher() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.reliability_profile.default_autonomy_level = AutonomyLevel.MANUAL

    class _Loop:
        async def handle_task(self, task: Task):
            from intergrax.runtime.task.task import TaskResult, TaskState

            assert task.options.governance.autonomy_level is AutonomyLevel.MANUAL
            return TaskResult(task_id=task.task_id, state=TaskState.COMPLETED, answer="ok")

    enricher = build_reliability_task_enricher(env)
    runner = build_task_runner_with_enricher(_Loop(), enricher)  # type: ignore[arg-type]

    await runner.run_task(
        Task(
            task_id="t",
            tenant_id="t",
            user_id="u",
            message="m",
        )
    )
