# © Artur Czarnecki. All rights reserved.

"""APP-EVOL-1 — EnvironmentSnapshot capture on task intake."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.environment_snapshot_wiring import (
    capture_environment_snapshot,
    compute_profile_snapshot_id,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_snapshot import ENV_SNAPSHOT_RUNTIME_KEY
from intergrax.applications.contracts.environment_state import (
    APP_ENV_STATE_RUNTIME_KEY,
    ApplicationEnvironmentState,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_capture_environment_snapshot_is_deterministic() -> None:
    manifest = ApplicationManifest.lab(
        app_id="snapshot_deterministic",
        name="Snapshot Deterministic",
        route_prefix="/v1/snapshot_deterministic",
        env_prefix="SNAPSHOT_DETERMINISTIC_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="snapshot_deterministic.lab")
    first = capture_environment_snapshot(manifest, environment)
    second = capture_environment_snapshot(manifest, environment)
    assert first.snapshot_id == second.snapshot_id
    assert first.profile_snapshot_id == second.profile_snapshot_id
    assert first.profile_snapshot_id.startswith("prof_")
    assert first.snapshot_id.startswith("envsnap_")


def test_profile_snapshot_id_changes_when_profile_changes() -> None:
    manifest = ApplicationManifest.lab(
        app_id="snapshot_profile_delta",
        name="Snapshot Profile Delta",
        route_prefix="/v1/snapshot_profile_delta",
        env_prefix="SNAPSHOT_PROFILE_DELTA_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="snapshot_profile_delta.lab")
    mutated = base.model_copy(update={"spec_version": "1.0.1"})
    assert compute_profile_snapshot_id(base) != compute_profile_snapshot_id(mutated)


def test_build_harness_host_runtime_mounts_snapshot_middleware() -> None:
    manifest = ApplicationManifest.lab(
        app_id="snapshot_middleware_wiring",
        name="Snapshot Middleware Wiring",
        route_prefix="/v1/snapshot_middleware_wiring",
        env_prefix="SNAPSHOT_MIDDLEWARE_WIRING_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="snapshot_middleware_wiring.lab")
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        use_in_memory_trace=True,
    )
    pipeline = resolve_harness_host_nexus_loop_legacy(runtime).middleware
    assert isinstance(pipeline, MiddlewarePipeline)
    names = [mw.name for mw in pipeline._middleware]  # noqa: SLF001
    assert "environment_snapshot" in names
    assert names.index("environment_snapshot") < names.index("application_environment_state")


@pytest.mark.asyncio
async def test_strict_intake_records_profile_snapshot_id() -> None:
    manifest = ApplicationManifest.lab(
        app_id="snapshot_strict_intake",
        name="Snapshot Strict Intake",
        route_prefix="/v1/snapshot_strict_intake",
        env_prefix="SNAPSHOT_STRICT_INTAKE_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(
        profile_id="snapshot_strict_intake.lab",
    ).model_copy(update={"execution_mode": ExecutionMode.STRICT})
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        use_in_memory_trace=True,
    )
    task = Task(
        task_id="task-strict-snapshot-1",
        tenant_id="tenant-test",
        user_id="user-test",
        message="strict intake",
        context=TaskContext(capability="echo.basic"),
    )
    coordinator = resolve_harness_host_nexus_loop_legacy(runtime)._lifecycle_hooks  # noqa: SLF001
    await coordinator.before(
        HookPoint.BEFORE_TASK_INTAKE,
        task,
        phase=ExecutionPhase.INTAKE,
    )

    snapshot = task.metadata.get(ENV_SNAPSHOT_RUNTIME_KEY)
    assert isinstance(snapshot, dict)
    assert snapshot["profile_snapshot_id"].startswith("prof_")

    env_state = ApplicationEnvironmentState.model_validate(task.metadata[APP_ENV_STATE_RUNTIME_KEY])
    assert env_state.profile_snapshot_id == snapshot["profile_snapshot_id"]
    assert env_state.profile_snapshot_id != environment.profile_id
