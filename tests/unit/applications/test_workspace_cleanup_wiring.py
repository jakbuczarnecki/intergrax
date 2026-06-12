# © Artur Czarnecki. All rights reserved.

"""APP-CON-8 / APP-PROD-8 — workspace cleanup wiring and env-state isolation refs."""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.workspace_cleanup_wiring import (
    apply_factory_lifespans,
    build_factory_lifespans,
    make_workspace_cleanup_lifespan,
    purge_all_workspace_sessions,
    sync_isolation_refs_for_hook,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_state import (
    APP_ENV_STATE_RUNTIME_KEY,
    ApplicationEnvironmentState,
    seed_application_environment_state,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.hooks.hook_context import HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_SESSION_ID_KEY
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.workspace.shadow_workspace import SHADOW_WORKSPACE_ID_KEY

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_sync_isolation_refs_for_hook_populates_env_state(tmp_path) -> None:
    shadow_manager = ShadowWorkspaceManager(root=tmp_path / "shadow")
    sandbox_manager = SandboxSessionManager(root=tmp_path / "sandbox")
    workspace = shadow_manager.open_or_create(tenant_id="tenant-1", task_id="task-1")
    session = sandbox_manager.open_or_create(tenant_id="tenant-1", task_id="task-1")
    seeded = seed_application_environment_state(
        app_id="legal",
        profile_id="legal.product",
        execution_mode=ExecutionMode.STRICT,
        task_id="task-1",
    )
    state = ApplicationEnvironmentState.model_validate(seeded[APP_ENV_STATE_RUNTIME_KEY])
    ctx = HookContext(
        task_id="task-1",
        run_id="task-1",
        runtime_state={
            "tenant_id": "tenant-1",
            SHADOW_WORKSPACE_ID_KEY: workspace.workspace_id,
            SANDBOX_SESSION_ID_KEY: session.session_id,
        },
    )
    updated = sync_isolation_refs_for_hook(
        ctx,
        state,
        shadow_manager=shadow_manager,
        sandbox_manager=sandbox_manager,
    )
    assert updated.shadow_workspace is not None
    assert updated.shadow_workspace.workspace_id == workspace.workspace_id
    assert updated.shadow_workspace.root_path == str(workspace.root)
    assert updated.sandbox_session is not None
    assert updated.sandbox_session.session_id == session.session_id


@pytest.mark.asyncio
async def test_workspace_cleanup_lifespan_disposes_active_sessions(tmp_path) -> None:
    shadow_manager = ShadowWorkspaceManager(root=tmp_path / "shadow")
    sandbox_manager = SandboxSessionManager(root=tmp_path / "sandbox")
    shadow_manager.open_or_create(tenant_id="tenant-1", task_id="task-1")
    sandbox_manager.open_or_create(tenant_id="tenant-1", task_id="task-1")
    assert shadow_manager.active_count == 1
    assert sandbox_manager.active_count == 1

    app = FastAPI()
    from intergrax.applications._shared.fastapi_mcp import apply_lifespans

    cleanup_lifespan = make_workspace_cleanup_lifespan(shadow_manager, sandbox_manager)
    apply_lifespans(app, cleanup_lifespan)
    async with app.router.lifespan_context(app):
        assert shadow_manager.active_count == 1
    assert shadow_manager.active_count == 0
    assert sandbox_manager.active_count == 0


def test_purge_all_workspace_sessions(tmp_path) -> None:
    shadow_manager = ShadowWorkspaceManager(root=tmp_path / "shadow")
    sandbox_manager = SandboxSessionManager(root=tmp_path / "sandbox")
    shadow_manager.open_or_create(tenant_id="tenant-1", task_id="task-1")
    sandbox_manager.open_or_create(tenant_id="tenant-1", task_id="task-1")
    shadow_disposed, sandbox_disposed = purge_all_workspace_sessions(
        shadow_manager=shadow_manager,
        sandbox_manager=sandbox_manager,
    )
    assert shadow_disposed == 1
    assert sandbox_disposed == 1
    assert shadow_manager.active_count == 0
    assert sandbox_manager.active_count == 0


@pytest.mark.asyncio
async def test_harness_factory_lifespan_teardown(tmp_path) -> None:
    manifest = ApplicationManifest.lab(
        app_id="workspace_cleanup_factory_test",
        name="Workspace Cleanup Factory Test",
        route_prefix="/v1/workspace_cleanup_factory_test",
        env_prefix="WORKSPACE_CLEANUP_FACTORY_TEST_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(
        profile_id="workspace_cleanup_factory_test.lab"
    )
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        use_in_memory_trace=True,
    )
    shadow_manager = runtime.env_wiring.shadow_manager
    sandbox_manager = runtime.env_wiring.sandbox_manager
    assert shadow_manager is not None
    assert sandbox_manager is not None
    shadow_manager.open_or_create(tenant_id="tenant-1", task_id="task-orphan")
    sandbox_manager.open_or_create(tenant_id="tenant-1", task_id="task-orphan")

    app = FastAPI()
    apply_factory_lifespans(app, runtime)
    async with app.router.lifespan_context(app):
        pass
    assert shadow_manager.active_count == 0
    assert sandbox_manager.active_count == 0


@pytest.mark.asyncio
async def test_environment_state_middleware_syncs_isolation_refs(tmp_path) -> None:
    manifest = ApplicationManifest.lab(
        app_id="env_state_isolation_test",
        name="Env State Isolation Test",
        route_prefix="/v1/env_state_isolation_test",
        env_prefix="ENV_STATE_ISOLATION_TEST_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="env_state_isolation_test.lab")
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        use_in_memory_trace=True,
    )
    shadow_manager = runtime.env_wiring.shadow_manager
    assert shadow_manager is not None
    workspace = shadow_manager.open_or_create(tenant_id="tenant-test", task_id="task-iso-1")
    task = Task(
        task_id="task-iso-1",
        tenant_id="tenant-test",
        user_id="user-test",
        message="hello",
        context=TaskContext(capability="echo.basic"),
    )
    coordinator = runtime.nexus_loop._lifecycle_hooks  # noqa: SLF001
    await coordinator.before(
        HookPoint.BEFORE_AGENT_SELECTION,
        task,
        phase=ExecutionPhase.AGENT_SELECTION,
        extra={SHADOW_WORKSPACE_ID_KEY: workspace.workspace_id},
    )
    persisted = task.metadata.get(APP_ENV_STATE_RUNTIME_KEY)
    assert isinstance(persisted, dict)
    state = ApplicationEnvironmentState.model_validate(persisted)
    assert state.shadow_workspace is not None
    assert state.shadow_workspace.workspace_id == workspace.workspace_id


def test_build_factory_lifespans_includes_workspace_cleanup() -> None:
    manifest = ApplicationManifest.lab(
        app_id="factory_lifespan_test",
        name="Factory Lifespan Test",
        route_prefix="/v1/factory_lifespan_test",
        env_prefix="FACTORY_LIFESPAN_TEST_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="factory_lifespan_test.lab")
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        use_in_memory_trace=True,
    )
    lifespans = build_factory_lifespans(runtime)
    assert len(lifespans) >= 1
