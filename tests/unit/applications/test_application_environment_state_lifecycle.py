# © Artur Czarnecki. All rights reserved.

"""APP-CON-3 — ApplicationEnvironmentState lifecycle sync on Nexus hooks."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_state import (
    APP_ENV_STATE_RUNTIME_KEY,
    ApplicationEnvironmentState,
    EnvironmentHealthStatus,
    EnvironmentTaskPhase,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.harness.application_host import ApplicationHost
from intergrax.runtime.hooks.hook_context import HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class _PhaseCaptureHost(ApplicationHost):
    def __init__(self) -> None:
        self.phases: list[tuple[HookPoint, EnvironmentTaskPhase]] = []

    def on_hook(self, point: HookPoint, context: HookContext) -> HookResult | None:
        state = ApplicationEnvironmentState.from_runtime_state(context.runtime_state)
        if state is not None:
            self.phases.append((point, state.phase))
        return None


def test_build_harness_host_runtime_mounts_environment_state_middleware() -> None:
    manifest = ApplicationManifest.lab(
        app_id="env_state_wiring_test",
        name="Env State Wiring Test",
        route_prefix="/v1/env_state_wiring_test",
        env_prefix="ENV_STATE_WIRING_TEST_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="env_state_wiring_test.lab")
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        use_in_memory_trace=True,
    )
    pipeline = resolve_harness_host_nexus_loop_legacy(runtime).middleware
    assert isinstance(pipeline, MiddlewarePipeline)
    names = [mw.name for mw in pipeline._middleware]  # noqa: SLF001
    assert "application_environment_state" in names


@pytest.mark.asyncio
async def test_environment_state_phase_tracks_lifecycle_hooks() -> None:
    manifest = ApplicationManifest.lab(
        app_id="env_state_lifecycle_test",
        name="Env State Lifecycle Test",
        route_prefix="/v1/env_state_lifecycle_test",
        env_prefix="ENV_STATE_LIFECYCLE_TEST_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="env_state_lifecycle_test.lab")
    host = _PhaseCaptureHost()
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        use_in_memory_trace=True,
        application_host=host,
    )
    task = Task(
        task_id="task-env-state-1",
        tenant_id="tenant-test",
        user_id="user-test",
        message="hello",
        context=TaskContext(capability="echo.basic"),
    )
    coordinator = resolve_harness_host_nexus_loop_legacy(runtime)._lifecycle_hooks  # noqa: SLF001

    await coordinator.before(
        HookPoint.BEFORE_TASK_INTAKE,
        task,
        phase=ExecutionPhase.INTAKE,
    )
    await coordinator.after(
        HookPoint.AFTER_TASK_INTAKE,
        task,
        phase=ExecutionPhase.INTAKE,
    )
    await coordinator.before(
        HookPoint.BEFORE_CLASSIFICATION,
        task,
        phase=ExecutionPhase.CLASSIFICATION,
    )

    assert host.phases[0] == (HookPoint.BEFORE_TASK_INTAKE, EnvironmentTaskPhase.INTAKE)
    assert host.phases[-1] == (HookPoint.BEFORE_CLASSIFICATION, EnvironmentTaskPhase.CLASSIFICATION)

    persisted = task.metadata.get(APP_ENV_STATE_RUNTIME_KEY)
    assert isinstance(persisted, dict)
    restored = ApplicationEnvironmentState.model_validate(persisted)
    assert restored.task_id == "task-env-state-1"
    assert restored.phase == EnvironmentTaskPhase.CLASSIFICATION
    assert restored.app_id == "env_state_lifecycle_test"


@pytest.mark.asyncio
async def test_environment_state_hitl_hook_updates_health() -> None:
    manifest = ApplicationManifest.lab(
        app_id="env_state_hitl_test",
        name="Env State HITL Test",
        route_prefix="/v1/env_state_hitl_test",
        env_prefix="ENV_STATE_HITL_TEST_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="env_state_hitl_test.lab")
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        use_in_memory_trace=True,
    )
    task = Task(
        task_id="task-env-hitl-1",
        tenant_id="tenant-test",
        user_id="user-test",
        message="approve me",
        context=TaskContext(capability="echo.basic"),
    )
    coordinator = resolve_harness_host_nexus_loop_legacy(runtime)._lifecycle_hooks  # noqa: SLF001

    await coordinator.before(
        HookPoint.BEFORE_HUMAN_APPROVAL,
        task,
        phase=ExecutionPhase.HUMAN_APPROVAL,
        extra={"hitl_ticket_id": "ticket-42", "hitl_reason": "policy_review"},
    )

    persisted = task.metadata.get(APP_ENV_STATE_RUNTIME_KEY)
    assert isinstance(persisted, dict)
    state = ApplicationEnvironmentState.model_validate(persisted)
    assert state.phase == EnvironmentTaskPhase.HITL
    assert state.health == EnvironmentHealthStatus.HITL_PENDING
    assert state.hitl.pending is True
    assert state.hitl.ticket_id == "ticket-42"
