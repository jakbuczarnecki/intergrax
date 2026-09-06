# © Artur Czarnecki. All rights reserved.

"""P1.6A — prepare-before-publish canonical host activation ordering."""

from __future__ import annotations

from unittest.mock import patch

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
    materialize_effective_profile_revision,
    pin_effective_profile_revision_for_execution,
    require_execution_pinned_revision,
    resolve_profile,
    resolve_revision_for_execution,
)
from intergrax.applications._shared.runtime_inspection.service import RuntimeInspectionService
from intergrax.applications.contracts.capability_dependency import (
    RequiredCapabilityDependencyUnavailableError,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import CostProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications.contracts.profile_resolution import (
    EffectiveProfileActivationConflictError,
    EffectiveProfileRevisionScope,
)
from intergrax.contracts.execution_identity import ExecutionId, mint_execution_id, mint_task_id
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry.profile import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_SCOPE = EffectiveProfileRevisionScope(application_id="p16a_host", tenant_id="tenant-a")


def _application(*, max_tool_calls: int | None = None) -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="p16a_host")
    if max_tool_calls is None:
        return profile
    return profile.model_copy(
        update={
            "governance": profile.governance.model_copy(
                update={"cost": CostProfile(max_tool_calls=max_tool_calls)},
            ),
        },
    )


def _echo_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="p16a_host",
        name="P16A Host",
        route_prefix="/v1/p16a_host",
        env_prefix="P16A_HOST_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )


def _echo_task() -> Task:
    return Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        message="p16a proof",
        context=TaskContext(capability="echo.basic"),
        agent_id="echo",
    )


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env, agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.harness_host_runtime.resolve_environment_llm_adapter",
        _resolve,
    )


def _build_runtime(
    application: ApplicationEnvironmentProfile,
    *,
    revision_store: InMemoryEffectiveProfileRevisionStore | None = None,
    pinning_store: InMemoryEffectiveProfileExecutionPinningStore | None = None,
    active_store: InMemoryActiveEffectiveProfileRevisionStore | None = None,
) -> object:
    return build_harness_host_runtime(
        _echo_manifest(),
        application,
        tenant_id="tenant-a",
        use_in_memory_trace=True,
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
    )


def _materialize(
    application: ApplicationEnvironmentProfile,
    revision_store: InMemoryEffectiveProfileRevisionStore,
) -> object:
    resolution = resolve_profile(application, layers=())
    return materialize_effective_profile_revision(
        resolution,
        scope=_SCOPE,
        store=revision_store,
    )


def test_environment_wiring_failure_leaves_active_unchanged() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    runtime_r1 = _build_runtime(
        _application(max_tool_calls=3),
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
    )
    r1_binding = active_store.get_active(_SCOPE)
    assert r1_binding is not None
    assert r1_binding.revision_id == runtime_r1.effective_profile_revision.revision_id

    with patch(
        "intergrax.applications._shared.harness_host_runtime.wire_application_environment",
        side_effect=RuntimeError("wire failed"),
    ):
        with pytest.raises(RuntimeError, match="wire failed"):
            _build_runtime(
                _application(max_tool_calls=7),
                revision_store=revision_store,
                pinning_store=pinning_store,
                active_store=active_store,
            )

    assert active_store.get_active(_SCOPE).revision_id == r1_binding.revision_id


def test_first_startup_environment_failure_leaves_active_none() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    with patch(
        "intergrax.applications._shared.harness_host_runtime.wire_application_environment",
        side_effect=RuntimeError("wire failed"),
    ):
        with pytest.raises(RuntimeError, match="wire failed"):
            _build_runtime(
                _application(),
                revision_store=revision_store,
                active_store=active_store,
            )
    assert active_store.get_active(_SCOPE) is None


def test_registry_preparation_failure_leaves_active_unchanged() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    runtime_r1 = _build_runtime(
        _application(max_tool_calls=2),
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
    )
    r1_binding = active_store.get_active(_SCOPE)

    with patch(
        "intergrax.applications._shared.harness_host_runtime.resolve_harness_host_registry",
        side_effect=RuntimeError("registry failed"),
    ):
        with pytest.raises(RuntimeError, match="registry failed"):
            _build_runtime(
                _application(max_tool_calls=8),
                revision_store=revision_store,
                pinning_store=pinning_store,
                active_store=active_store,
            )

    assert active_store.get_active(_SCOPE).revision_id == r1_binding.revision_id


def test_prepare_occurs_before_activation_cas(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    original_wire = build_harness_host_runtime.__globals__["wire_application_environment"]

    def _tracked_wire(*args: object, **kwargs: object) -> object:
        events.append("prepare")
        return original_wire(*args, **kwargs)

    original_activate = build_harness_host_runtime.__globals__["activate_materialized_revision"]

    def _tracked_activate(*args: object, **kwargs: object) -> object:
        events.append("activation_cas")
        return original_activate(*args, **kwargs)

    monkeypatch.setattr(
        "intergrax.applications._shared.harness_host_runtime.wire_application_environment",
        _tracked_wire,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.harness_host_runtime.activate_materialized_revision",
        _tracked_activate,
    )

    _build_runtime(_application())
    assert "prepare" in events
    assert "activation_cas" in events
    assert events.index("prepare") < events.index("activation_cas")


def test_successful_builder_performs_exactly_one_cas() -> None:
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    cas_calls = 0
    original_cas = active_store.compare_and_set_active

    def _tracked_cas(*args: object, **kwargs: object) -> object:
        nonlocal cas_calls
        cas_calls += 1
        return original_cas(*args, **kwargs)

    active_store.compare_and_set_active = _tracked_cas  # type: ignore[method-assign]
    _build_runtime(_application(), active_store=active_store)
    assert cas_calls == 1


def test_cas_conflict_after_preparation_fails_without_runtime() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    runtime_r1 = _build_runtime(
        _application(max_tool_calls=1),
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
    )
    r1_id = runtime_r1.effective_profile_revision.revision_id
    revision_r3 = _materialize(_application(max_tool_calls=99), revision_store)
    activation_service = EffectiveProfileActivationService(
        EffectiveProfileActivationDependencies(
            revision_store=revision_store,
            active_store=active_store,
        ),
    )
    activate_materialized_revision(
        activation_service,
        scope=_SCOPE,
        candidate_revision_id=revision_r3.revision_id,
    )
    r3_binding = active_store.get_active(_SCOPE)
    assert r3_binding is not None
    assert r3_binding.revision_id == revision_r3.revision_id
    assert r3_binding.revision_id != r1_id

    from intergrax.applications.contracts.profile_resolution.activation import (
        ActiveEffectiveProfileRevisionCasOutcome,
        ActiveEffectiveProfileRevisionCasResult,
    )

    def _conflict_cas(*args: object, **kwargs: object) -> ActiveEffectiveProfileRevisionCasResult:
        return ActiveEffectiveProfileRevisionCasResult(
            outcome=ActiveEffectiveProfileRevisionCasOutcome.CONFLICT,
            current_binding=active_store.get_active(_SCOPE),
        )

    active_store.compare_and_set_active = _conflict_cas  # type: ignore[method-assign]

    with pytest.raises(EffectiveProfileActivationConflictError):
        _build_runtime(
            _application(max_tool_calls=5),
            revision_store=revision_store,
            pinning_store=pinning_store,
            active_store=active_store,
        )

    assert active_store.get_active(_SCOPE).revision_id == revision_r3.revision_id


def test_conflict_path_does_not_restore_previous_active() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    _build_runtime(_application(max_tool_calls=1), revision_store=revision_store, active_store=active_store)
    revision_r3 = _materialize(_application(max_tool_calls=99), revision_store)
    activation_service = EffectiveProfileActivationService(
        EffectiveProfileActivationDependencies(
            revision_store=revision_store,
            active_store=active_store,
        ),
    )
    current = active_store.get_active(_SCOPE)
    activate_materialized_revision(
        activation_service,
        scope=_SCOPE,
        candidate_revision_id=revision_r3.revision_id,
    )
    r3_id = active_store.get_active(_SCOPE).revision_id
    prior_r1_id = current.revision_id

    from intergrax.applications.contracts.profile_resolution.activation import (
        ActiveEffectiveProfileRevisionCasOutcome,
        ActiveEffectiveProfileRevisionCasResult,
    )

    active_store.compare_and_set_active = lambda *a, **k: ActiveEffectiveProfileRevisionCasResult(  # type: ignore[method-assign]
        outcome=ActiveEffectiveProfileRevisionCasOutcome.CONFLICT,
        current_binding=active_store.get_active(_SCOPE),
    )

    with pytest.raises(EffectiveProfileActivationConflictError):
        _build_runtime(
            _application(max_tool_calls=5),
            revision_store=revision_store,
            active_store=active_store,
        )

    assert active_store.get_active(_SCOPE).revision_id == r3_id
    assert active_store.get_active(_SCOPE).revision_id != prior_r1_id


def test_inactive_materialized_revision_after_preparation_failure() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    runtime_r1 = _build_runtime(
        _application(max_tool_calls=2),
        revision_store=revision_store,
        active_store=active_store,
    )
    r1_id = runtime_r1.effective_profile_revision.revision_id

    with patch(
        "intergrax.applications._shared.harness_host_runtime.wire_application_environment",
        side_effect=RuntimeError("wire failed"),
    ):
        with pytest.raises(RuntimeError):
            _build_runtime(
                _application(max_tool_calls=6),
                revision_store=revision_store,
                active_store=active_store,
            )

    assert active_store.get_active(_SCOPE).revision_id == r1_id
    stored_ids = {revision.revision_id for revision in revision_store._revisions.values()}
    assert len(stored_ids) >= 2
    inactive = next(rid for rid in stored_ids if rid != r1_id)
    assert revision_store.get(inactive, scope=_SCOPE) is not None


@pytest.mark.asyncio
async def test_new_execution_after_failed_preparation_pins_old_active() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    runtime_r1 = _build_runtime(
        _application(max_tool_calls=2),
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
    )
    r1_id = runtime_r1.effective_profile_revision.revision_id

    with patch(
        "intergrax.applications._shared.harness_host_runtime.wire_application_environment",
        side_effect=RuntimeError("wire failed"),
    ):
        with pytest.raises(RuntimeError):
            _build_runtime(
                _application(max_tool_calls=6),
                revision_store=revision_store,
                pinning_store=pinning_store,
                active_store=active_store,
            )

    captured_execution_id: ExecutionId | None = None

    async def _capture_execute(self, request: object) -> TaskResult:
        from intergrax.contracts.execution_identity import require_active_execution_id

        nonlocal captured_execution_id
        captured_execution_id = require_active_execution_id()
        return TaskResult(
            task_id=_echo_task().task_id,
            state=TaskState.COMPLETED,
            answer="ok",
        )

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        _capture_execute,
    ):
        await runtime_r1.execution.execute(_echo_task())

    assert captured_execution_id is not None
    binding = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=captured_execution_id,
        pinning_store=pinning_store,
    )
    assert binding.revision_id == r1_id


@pytest.mark.asyncio
async def test_new_execution_after_successful_activation_pins_new_revision() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    _build_runtime(
        _application(max_tool_calls=2),
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
    )
    runtime_r2 = _build_runtime(
        _application(max_tool_calls=9),
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
    )
    r2_id = runtime_r2.effective_profile_revision.revision_id
    captured_execution_id: ExecutionId | None = None

    async def _capture_execute(self, request: object) -> TaskResult:
        from intergrax.contracts.execution_identity import require_active_execution_id

        nonlocal captured_execution_id
        captured_execution_id = require_active_execution_id()
        return TaskResult(
            task_id=_echo_task().task_id,
            state=TaskState.COMPLETED,
            answer="ok",
        )

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        _capture_execute,
    ):
        await runtime_r2.execution.execute(_echo_task())

    assert captured_execution_id is not None
    binding = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=captured_execution_id,
        pinning_store=pinning_store,
    )
    assert binding.revision_id == r2_id


@pytest.mark.asyncio
async def test_existing_execution_remains_pinned_to_r1() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    runtime_r1 = _build_runtime(
        _application(max_tool_calls=3),
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
    )
    runtime_r2 = _build_runtime(
        _application(max_tool_calls=7),
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
    )
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=runtime_r1.effective_profile_revision,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
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


def test_dependency_validation_failure_before_cas_leaves_active_unchanged() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    runtime_r1 = _build_runtime(
        _application(max_tool_calls=2),
        revision_store=revision_store,
        active_store=active_store,
    )
    r1_id = runtime_r1.effective_profile_revision.revision_id
    invalid_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="p16a_host").model_copy(
        update={
            "tool_profile": ToolProfile(enabled=["rag.retrieve"]),
            "skill_profile": SkillProfile(enabled_bundles=["legal"]),
        },
    )
    with pytest.raises(RequiredCapabilityDependencyUnavailableError):
        _build_runtime(invalid_env, revision_store=revision_store, active_store=active_store)
    assert active_store.get_active(_SCOPE).revision_id == r1_id


def test_runtime_revision_matches_active_binding_after_success() -> None:
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    runtime = _build_runtime(_application(), active_store=active_store)
    binding = active_store.get_active(_SCOPE)
    assert binding is not None
    assert runtime.effective_profile_revision is not None
    assert runtime.effective_profile_revision.revision_id == binding.revision_id
    assert runtime.effective_profile_revision.fingerprint == binding.fingerprint
    assert runtime.effective_profile_revision.scope == binding.scope


def test_inspection_shows_old_active_before_cas_and_new_after() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    inspection = RuntimeInspectionService(
        revision_store=revision_store,
        active_store=active_store,
    )
    assert inspection.inspect_active_revision(scope=_SCOPE).revision is None

    runtime_r1 = _build_runtime(
        _application(max_tool_calls=2),
        revision_store=revision_store,
        active_store=active_store,
    )
    active_r1 = inspection.inspect_active_revision(scope=_SCOPE).revision
    assert active_r1 is not None
    assert active_r1.revision_id == runtime_r1.effective_profile_revision.revision_id

    runtime_r2 = _build_runtime(
        _application(max_tool_calls=8),
        revision_store=revision_store,
        active_store=active_store,
    )
    active_r2 = inspection.inspect_active_revision(scope=_SCOPE).revision
    assert active_r2 is not None
    assert active_r2.revision_id == runtime_r2.effective_profile_revision.revision_id
    assert active_r2.revision_id != active_r1.revision_id
