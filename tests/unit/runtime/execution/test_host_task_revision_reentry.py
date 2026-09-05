# © Artur Czarnecki. All rights reserved.

"""P1.2B — fail-closed re-entry when revision binding is missing."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.profile_resolution import (
    require_execution_pinned_revision,
    resolve_revision_for_execution,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import CostProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications.contracts.profile_resolution import (
    EffectiveProfileRevisionScope,
    MissingPinnedEffectiveProfileRevisionError,
)
from intergrax.applications._shared.profile_resolution import (
    InMemoryEffectiveProfileExecutionPinningStore,
    InMemoryEffectiveProfileRevisionStore,
)
from intergrax.contracts.execution_identity import (
    ExecutionId,
    mint_execution_id,
    mint_task_id,
    require_active_execution_id,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_SCOPE = EffectiveProfileRevisionScope(application_id="revision_reentry", tenant_id="tenant-a")


def _application(*, max_tool_calls: int | None = None) -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="revision_reentry")
    updates: dict[str, object] = {
        "meta": profile.meta.model_copy(update={"execution_mode": ExecutionMode.BALANCED}),
    }
    if max_tool_calls is not None:
        updates["governance"] = profile.governance.model_copy(
            update={"cost": CostProfile(max_tool_calls=max_tool_calls)},
        )
    return profile.model_copy(update=updates)


def _echo_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="revision_reentry",
        name="Revision Re-entry Host",
        route_prefix="/v1/revision_reentry",
        env_prefix="REVISION_REENTRY_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )


def _build_runtime(
    application: ApplicationEnvironmentProfile,
    *,
    revision_store: InMemoryEffectiveProfileRevisionStore,
    pinning_store: InMemoryEffectiveProfileExecutionPinningStore,
) -> object:
    return build_harness_host_runtime(
        _echo_manifest(),
        application,
        tenant_id="tenant-a",
        use_in_memory_trace=True,
        revision_store=revision_store,
        pinning_store=pinning_store,
    )


def _echo_task() -> Task:
    return Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        message="revision re-entry proof",
        context=TaskContext(capability="echo.basic"),
        agent_id="echo",
    )


@pytest.mark.asyncio
async def test_host_task_restore_missing_binding_fails_closed() -> None:
    runtime = _build_runtime(
        _application(),
        revision_store=InMemoryEffectiveProfileRevisionStore(),
        pinning_store=InMemoryEffectiveProfileExecutionPinningStore(),
    )
    execution_id = mint_execution_id()
    agent_spy = AsyncMock()
    orchestration_spy = AsyncMock()

    with (
        patch(
            "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
            agent_spy,
        ),
        patch(
            "intergrax.runtime.execution.host_task.TaskBoundOrchestrationDelegate.execute",
            orchestration_spy,
        ),
        patch.object(ActiveTaskRegistry, "register", new_callable=AsyncMock) as register_spy,
        pytest.raises(MissingPinnedEffectiveProfileRevisionError),
    ):
        await runtime.execution.execute(
            _echo_task(),
            execution_id=execution_id,
            restore_existing_execution=True,
        )

    agent_spy.assert_not_called()
    orchestration_spy.assert_not_called()
    register_spy.assert_not_called()


@pytest.mark.asyncio
async def test_host_task_new_explicit_execution_id_pins() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    runtime = _build_runtime(
        _application(max_tool_calls=5),
        revision_store=revision_store,
        pinning_store=pinning_store,
    )
    execution_id = mint_execution_id()

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=mint_task_id(),
            state=TaskState.COMPLETED,
            answer="ok",
        ),
    ):
        await runtime.execution.execute(
            _echo_task(),
            execution_id=execution_id,
            restore_existing_execution=False,
        )

    binding = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
    )
    assert binding.revision_id == runtime.effective_profile_revision.revision_id


@pytest.mark.asyncio
async def test_host_task_existing_binding_idempotent_restore() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    runtime = _build_runtime(
        _application(max_tool_calls=3),
        revision_store=revision_store,
        pinning_store=pinning_store,
    )
    task = _echo_task()
    execution_id = mint_execution_id()

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(task_id=task.task_id, state=TaskState.COMPLETED, answer="ok"),
    ):
        await runtime.execution.execute(task, execution_id=execution_id)

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(task_id=task.task_id, state=TaskState.COMPLETED, answer="ok"),
    ):
        await runtime.execution.execute(
            _echo_task(),
            execution_id=execution_id,
            restore_existing_execution=True,
        )

    binding = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
    )
    assert binding.revision_id == runtime.effective_profile_revision.revision_id


@pytest.mark.asyncio
async def test_host_task_r1_preserved_under_r2_restore() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    runtime_r1 = _build_runtime(
        _application(max_tool_calls=2),
        revision_store=revision_store,
        pinning_store=pinning_store,
    )
    runtime_r2 = _build_runtime(
        _application(max_tool_calls=9),
        revision_store=revision_store,
        pinning_store=pinning_store,
    )
    task = _echo_task()
    execution_id = mint_execution_id()

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(task_id=task.task_id, state=TaskState.COMPLETED, answer="ok"),
    ):
        await runtime_r1.execution.execute(task, execution_id=execution_id)

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(task_id=task.task_id, state=TaskState.COMPLETED, answer="ok"),
    ):
        await runtime_r2.execution.execute(
            _echo_task(),
            execution_id=execution_id,
            restore_existing_execution=True,
        )

    resolved = resolve_revision_for_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
        scope_application_id=_SCOPE.application_id,
        scope_tenant_id=_SCOPE.tenant_id,
    )
    assert resolved.revision_id == runtime_r1.effective_profile_revision.revision_id
    assert resolved.revision_id != runtime_r2.effective_profile_revision.revision_id


@pytest.mark.asyncio
async def test_host_task_missing_binding_never_pins_r2() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    runtime_r1 = _build_runtime(
        _application(max_tool_calls=2),
        revision_store=revision_store,
        pinning_store=pinning_store,
    )
    runtime_r2 = _build_runtime(
        _application(max_tool_calls=9),
        revision_store=revision_store,
        pinning_store=pinning_store,
    )
    execution_id = mint_execution_id()

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=mint_task_id(),
            state=TaskState.COMPLETED,
            answer="ok",
        ),
    ):
        await runtime_r1.execution.execute(_echo_task(), execution_id=execution_id)

    pinning_store._bindings.clear()

    with pytest.raises(MissingPinnedEffectiveProfileRevisionError):
        await runtime_r2.execution.execute(
            _echo_task(),
            execution_id=execution_id,
            restore_existing_execution=True,
        )

    assert pinning_store.get(tenant_id="tenant-a", execution_id=execution_id) is None


@pytest.mark.asyncio
async def test_resume_implicit_reentry_missing_binding_fails_closed() -> None:
    runtime = _build_runtime(
        _application(),
        revision_store=InMemoryEffectiveProfileRevisionStore(),
        pinning_store=InMemoryEffectiveProfileExecutionPinningStore(),
    )
    task = _echo_task()
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id="tenant-a",
        resume_token="resume-implicit-reentry",
        task_state=task.state,
        task_snapshot=task.model_dump(mode="json"),
    )
    with pytest.raises(MissingPinnedEffectiveProfileRevisionError):
        await runtime.execution.execute(
            task,
            execution_id=mint_execution_id(),
            resume_checkpoint=checkpoint,
            restore_existing_execution=False,
        )


@pytest.mark.asyncio
async def test_missing_binding_reentry_no_meaningful_work() -> None:
    runtime = _build_runtime(
        _application(),
        revision_store=InMemoryEffectiveProfileRevisionStore(),
        pinning_store=InMemoryEffectiveProfileExecutionPinningStore(),
    )
    captured_execution_id: ExecutionId | None = None

    async def _should_not_run(self, request):
        nonlocal captured_execution_id
        captured_execution_id = require_active_execution_id()
        return TaskResult(task_id=mint_task_id(), state=TaskState.COMPLETED, answer="ok")

    with (
        patch(
            "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
            _should_not_run,
        ),
        patch(
            "intergrax.runtime.execution.host_task.TaskBoundOrchestrationDelegate.execute",
            new_callable=AsyncMock,
        ) as orchestration_spy,
        patch.object(ActiveTaskRegistry, "register", new_callable=AsyncMock) as register_spy,
        pytest.raises(MissingPinnedEffectiveProfileRevisionError),
    ):
        await runtime.execution.execute(
            _echo_task(),
            execution_id=mint_execution_id(),
            restore_existing_execution=True,
        )

    assert captured_execution_id is None
    orchestration_spy.assert_not_called()
    register_spy.assert_not_called()
