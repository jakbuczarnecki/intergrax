# © Artur Czarnecki. All rights reserved.

"""P1.2A — canonical effective profile revision pinning adoption."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.profile_resolution import (
    EffectiveProfileActivationDependencies,
    EffectiveProfileActivationService,
    EffectiveProfileExecutionPinningDependencies,
    InMemoryActiveEffectiveProfileRevisionStore,
    InMemoryEffectiveProfileExecutionPinningStore,
    InMemoryEffectiveProfileRevisionStore,
    activate_materialized_revision,
    attach_revision_checkpoint_evidence_to_task,
    build_effective_profile_revision_admission,
    materialize_effective_profile_revision,
    pin_effective_profile_revision_for_execution,
    require_execution_pinned_revision,
    resolve_profile,
    resolve_revision_for_execution,
)
from intergrax.applications._shared.profile_resolution.persistence import (
    KvEffectiveProfileExecutionPinningStore,
    KvEffectiveProfileRevisionStore,
    wire_effective_profile_execution_pinning_store,
    wire_effective_profile_revision_store,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import CostProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications.contracts.profile_resolution import (
    EffectiveProfileRevisionConflictError,
    EffectiveProfileRevisionError,
    EffectiveProfileRevisionScope,
    MissingPinnedEffectiveProfileRevisionError,
    ProfileDelta,
    ProfileFieldUpdate,
    ProfileLayer,
    ProfileLayerInput,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionBinding,
)
from intergrax.contracts.execution_identity import (
    ExecutionId,
    mint_execution_id,
    mint_task_id,
    require_active_execution_id,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from tests.unit.runtime.background_execution.reentry_admission_doubles import InMemoryKVStore

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HARNESS_RUNTIME_PATH = _REPO_ROOT / "intergrax" / "applications" / "_shared" / "harness_host_runtime.py"
_SCOPE = EffectiveProfileRevisionScope(application_id="revision_adoption", tenant_id="tenant-a")


def _application(
    *,
    max_tool_calls: int | None = None,
    execution_mode: ExecutionMode = ExecutionMode.BALANCED,
) -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="revision_adoption")
    updates: dict[str, object] = {
        "meta": profile.meta.model_copy(update={"execution_mode": execution_mode}),
    }
    if max_tool_calls is not None:
        updates["governance"] = profile.governance.model_copy(
            update={"cost": CostProfile(max_tool_calls=max_tool_calls)},
        )
    return profile.model_copy(update=updates)


def _echo_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="revision_adoption",
        name="Revision Adoption Host",
        route_prefix="/v1/revision_adoption",
        env_prefix="REVISION_ADOPTION_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )


def _pinning_dependencies(
    revision: object,
    revision_store: InMemoryEffectiveProfileRevisionStore,
    pinning_store: InMemoryEffectiveProfileExecutionPinningStore,
) -> EffectiveProfileExecutionPinningDependencies:
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    activation_service = EffectiveProfileActivationService(
        EffectiveProfileActivationDependencies(
            revision_store=revision_store,
            active_store=active_store,
        ),
    )
    activate_materialized_revision(
        activation_service,
        scope=_SCOPE,
        candidate_revision_id=revision.revision_id,
    )
    return EffectiveProfileExecutionPinningDependencies(
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
        scope=_SCOPE,
    )


def _revision_from_application(
    application: ApplicationEnvironmentProfile,
    *,
    store: InMemoryEffectiveProfileRevisionStore | KvEffectiveProfileRevisionStore,
    layers: tuple[ProfileLayerInput, ...] = (),
) -> object:
    resolution = resolve_profile(application, layers=layers)
    return materialize_effective_profile_revision(
        resolution,
        scope=_SCOPE,
        store=store,
    )


def _build_runtime(
    application: ApplicationEnvironmentProfile,
    *,
    revision_store: InMemoryEffectiveProfileRevisionStore | KvEffectiveProfileRevisionStore | None = None,
    pinning_store: InMemoryEffectiveProfileExecutionPinningStore
    | KvEffectiveProfileExecutionPinningStore
    | None = None,
    kv_store: InMemoryKVStore | None = None,
) -> object:
    return build_harness_host_runtime(
        _echo_manifest(),
        application,
        tenant_id="tenant-a",
        use_in_memory_trace=True,
        revision_store=revision_store,
        pinning_store=pinning_store,
        key_value_cache=kv_store,
    )


def _echo_task() -> Task:
    return Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        message="revision adoption proof",
        context=TaskContext(capability="echo.basic"),
        agent_id="echo",
    )


@pytest.mark.asyncio
async def test_canonical_harness_execution_pins_revision_before_work() -> None:
    runtime = _build_runtime(_application())
    task = _echo_task()
    captured_execution_id: ExecutionId | None = None

    async def _capture_execute(self, request):
        nonlocal captured_execution_id
        captured_execution_id = require_active_execution_id()
        return TaskResult(
            task_id=task.task_id,
            state=TaskState.COMPLETED,
            answer="ok",
        )

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        _capture_execute,
    ):
        await runtime.execution.execute(task)

    assert captured_execution_id is not None
    binding = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=captured_execution_id,
        pinning_store=runtime.effective_profile_pinning_store,
    )
    assert binding.revision_id == runtime.effective_profile_revision.revision_id
    assert runtime.execution._revision_admission is not None


def test_harness_builder_supplies_revision_admission_dependency() -> None:
    source = _HARNESS_RUNTIME_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    build_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_environment_host_task_execution"
    )
    keyword_names = {keyword.arg for keyword in build_call.keywords}
    assert "pinning_dependencies" in keyword_names


@pytest.mark.asyncio
async def test_r1_then_r2_old_execution_stays_on_r1() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    runtime_r1 = _build_runtime(
        _application(max_tool_calls=3),
        revision_store=revision_store,
        pinning_store=pinning_store,
    )
    runtime_r2 = _build_runtime(
        _application(max_tool_calls=7),
        revision_store=revision_store,
        pinning_store=pinning_store,
    )
    assert runtime_r1.effective_profile_revision.revision_id != (
        runtime_r2.effective_profile_revision.revision_id
    )

    task = _echo_task()
    captured_execution_id: ExecutionId | None = None

    async def _capture_execute(self, request):
        nonlocal captured_execution_id
        captured_execution_id = require_active_execution_id()
        return TaskResult(task_id=task.task_id, state=TaskState.COMPLETED, answer="ok")

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        _capture_execute,
    ):
        await runtime_r1.execution.execute(task)

    assert captured_execution_id is not None
    execution_id = captured_execution_id

    binding = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
    )
    assert binding.revision_id == runtime_r1.effective_profile_revision.revision_id

    resumed_task = _echo_task()
    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=resumed_task.task_id,
            state=TaskState.COMPLETED,
            answer="ok",
        ),
    ):
        await runtime_r2.execution.execute(resumed_task, execution_id=execution_id)

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
async def test_new_execution_after_r2_uses_r2() -> None:
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
    captured_execution_id: ExecutionId | None = None

    async def _capture_execute(self, request):
        nonlocal captured_execution_id
        captured_execution_id = require_active_execution_id()
        return TaskResult(task_id=task.task_id, state=TaskState.COMPLETED, answer="ok")

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        _capture_execute,
    ):
        await runtime_r2.execution.execute(task)

    assert captured_execution_id is not None
    binding = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=captured_execution_id,
        pinning_store=pinning_store,
    )
    assert binding.revision_id == runtime_r2.effective_profile_revision.revision_id


@pytest.mark.asyncio
async def test_missing_binding_on_resume_fails_closed() -> None:
    runtime = _build_runtime(_application())
    task = _echo_task()
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id="tenant-a",
        resume_token="resume-missing-binding",
        task_state=task.state,
        task_snapshot=task.model_dump(mode="json"),
    )
    with pytest.raises(MissingPinnedEffectiveProfileRevisionError):
        await runtime.execution.execute(
            task,
            execution_id=mint_execution_id(),
            resume_checkpoint=checkpoint,
        )


def test_conflicting_binding_fails_closed() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    revision_a = _revision_from_application(_application(max_tool_calls=1), store=revision_store)
    revision_b = _revision_from_application(_application(max_tool_calls=2), store=revision_store)
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision_a,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    with pytest.raises(EffectiveProfileRevisionConflictError):
        pin_effective_profile_revision_for_execution(
            revision=revision_b,
            tenant_id="tenant-a",
            execution_id=execution_id,
            pinning_store=pinning_store,
            revision_store=revision_store,
        )


def test_missing_revision_snapshot_fails_closed() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    revision = _revision_from_application(_application(), store=revision_store)
    execution_id = mint_execution_id()
    pinning_store.pin(
        EffectiveProfileExecutionBinding(
            tenant_id="tenant-a",
            execution_id=execution_id,
            revision_id=revision.revision_id,
            fingerprint=revision.fingerprint,
        )
    )
    revision_store._revisions.clear()
    with pytest.raises(MissingPinnedEffectiveProfileRevisionError):
        resolve_revision_for_execution(
            tenant_id="tenant-a",
            execution_id=execution_id,
            pinning_store=pinning_store,
            revision_store=revision_store,
            scope_application_id=_SCOPE.application_id,
            scope_tenant_id=_SCOPE.tenant_id,
        )


def test_fingerprint_mismatch_fails_closed() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    revision = _revision_from_application(_application(), store=revision_store)
    execution_id = mint_execution_id()
    pinning_store.pin(
        EffectiveProfileExecutionBinding(
            tenant_id="tenant-a",
            execution_id=execution_id,
            revision_id=revision.revision_id,
            fingerprint="mismatched-fingerprint",
        )
    )
    with pytest.raises(EffectiveProfileRevisionError, match="fingerprint mismatch"):
        resolve_revision_for_execution(
            tenant_id="tenant-a",
            execution_id=execution_id,
            pinning_store=pinning_store,
            revision_store=revision_store,
            scope_application_id=_SCOPE.application_id,
            scope_tenant_id=_SCOPE.tenant_id,
        )


@pytest.mark.asyncio
async def test_resume_preserves_pinned_revision_not_current_host_revision() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    runtime_r1 = _build_runtime(
        _application(max_tool_calls=4),
        revision_store=revision_store,
        pinning_store=pinning_store,
    )
    runtime_r2 = _build_runtime(
        _application(max_tool_calls=8),
        revision_store=revision_store,
        pinning_store=pinning_store,
    )
    task = _echo_task()
    captured_execution_id: ExecutionId | None = None

    async def _capture_first(self, request):
        nonlocal captured_execution_id
        captured_execution_id = require_active_execution_id()
        return TaskResult(task_id=task.task_id, state=TaskState.COMPLETED, answer="ok")

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        _capture_first,
    ):
        await runtime_r1.execution.execute(task)

    assert captured_execution_id is not None
    execution_id = captured_execution_id

    admitted_task = attach_revision_checkpoint_evidence_to_task(
        _echo_task(),
        runtime_r1.effective_profile_revision,
    )
    checkpoint = TaskCheckpoint(
        task_id=admitted_task.task_id,
        tenant_id="tenant-a",
        resume_token="resume-1",
        task_state=admitted_task.state,
        task_snapshot=admitted_task.model_dump(mode="json"),
    )
    resumed_task = _echo_task()
    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=resumed_task.task_id,
            state=TaskState.COMPLETED,
            answer="ok",
        ),
    ):
        await runtime_r2.execution.execute(
            resumed_task,
            execution_id=execution_id,
            resume_checkpoint=checkpoint,
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


def test_restore_existing_execution_missing_binding_fails_closed() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    admission = build_effective_profile_revision_admission(
        _pinning_dependencies(
            _revision_from_application(_application(), store=revision_store),
            revision_store,
            pinning_store,
        )
    )
    execution_id = mint_execution_id()
    with pytest.raises(MissingPinnedEffectiveProfileRevisionError):
        admission.admit_root_execution(
            tenant_id="tenant-a",
            execution_id=execution_id,
            task=_echo_task(),
            restore_existing_execution=True,
        )


@pytest.mark.asyncio
async def test_checkpoint_binding_mismatch_fails_closed() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    revision_a = _revision_from_application(_application(max_tool_calls=1), store=revision_store)
    revision_b = _revision_from_application(_application(max_tool_calls=2), store=revision_store)
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision_a,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    admitted_task = attach_revision_checkpoint_evidence_to_task(_echo_task(), revision_b)
    checkpoint = TaskCheckpoint(
        task_id=admitted_task.task_id,
        tenant_id="tenant-a",
        resume_token="resume-mismatch",
        task_state=admitted_task.state,
        task_snapshot=admitted_task.model_dump(mode="json"),
    )
    admission = build_effective_profile_revision_admission(
        _pinning_dependencies(revision_b, revision_store, pinning_store),
    )
    with pytest.raises(EffectiveProfileRevisionConflictError):
        admission.admit_root_execution(
            tenant_id="tenant-a",
            execution_id=execution_id,
            task=_echo_task(),
            resume_checkpoint=checkpoint,
        )


def test_durable_redelivery_preserves_revision_with_fresh_adapters() -> None:
    backing = InMemoryKVStore()
    revision_store_a = wire_effective_profile_revision_store(kv_store=backing)
    pinning_store_a = wire_effective_profile_execution_pinning_store(kv_store=backing)
    revision = _revision_from_application(_application(), store=revision_store_a)
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store_a,
        revision_store=revision_store_a,
    )

    revision_store_b = wire_effective_profile_revision_store(kv_store=backing)
    pinning_store_b = wire_effective_profile_execution_pinning_store(kv_store=backing)
    resolved = resolve_revision_for_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store_b,
        revision_store=revision_store_b,
        scope_application_id=_SCOPE.application_id,
        scope_tenant_id=_SCOPE.tenant_id,
    )
    assert resolved.revision_id == revision.revision_id
    assert revision_store_b is not revision_store_a
    assert pinning_store_b is not pinning_store_a


def test_tenant_scope_mismatch_fails_closed() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    revision = _revision_from_application(_application(), store=revision_store)
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    with pytest.raises(MissingPinnedEffectiveProfileRevisionError):
        resolve_revision_for_execution(
            tenant_id="tenant-b",
            execution_id=execution_id,
            pinning_store=pinning_store,
            revision_store=revision_store,
            scope_application_id=_SCOPE.application_id,
            scope_tenant_id="tenant-b",
        )


def test_production_strict_requires_durable_pinning_stores() -> None:
    with pytest.raises(EffectiveProfileRevisionError, match="durable effective profile revision store"):
        build_harness_host_runtime(
            _echo_manifest(),
            _application(execution_mode=ExecutionMode.STRICT),
            tenant_id="tenant-a",
            use_in_memory_trace=True,
        )
