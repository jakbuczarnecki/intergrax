# © Artur Czarnecki. All rights reserved.

"""P1.2 — effective profile revisioning and semantic diff."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.profile_resolution import (
    InMemoryEffectiveProfileExecutionPinningStore,
    InMemoryEffectiveProfileRevisionStore,
    attach_revision_checkpoint_evidence_to_task,
    diff_effective_profile_revisions,
    inherit_child_execution_pinned_revision,
    materialize_effective_profile_revision,
    pin_effective_profile_revision_for_execution,
    require_execution_pinned_revision,
    resolve_profile,
    resolve_revision_for_execution,
    revision_id_from_checkpoint,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionBinding,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import CostProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications.contracts.profile_resolution import (
    EffectiveProfileRevisionScope,
    MissingPinnedEffectiveProfileRevisionError,
    ProfileDelta,
    ProfileDiffChangeKind,
    ProfileFieldUpdate,
    ProfileLayer,
    ProfileLayerInput,
)
from intergrax.contracts.execution_identity import ExecutionId, mint_execution_id, mint_task_id
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.tools.registry.profile import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_SCOPE = EffectiveProfileRevisionScope(application_id="revision.test", tenant_id="tenant-a")


def _application(
    *,
    provider: LLMProvider = LLMProvider.OPENAI,
    model: str = "gpt-4o-mini",
    tools: list[str] | None = None,
    execution_mode: ExecutionMode = ExecutionMode.BALANCED,
    max_tool_calls: int | None = None,
) -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="revision.test")
    updates: dict[str, object] = {
        "meta": profile.meta.model_copy(update={"execution_mode": execution_mode}),
        "capabilities": profile.capabilities.model_copy(
            update={
                "llm": LLMProfile(provider=provider, model=model),
                "tools": ToolProfile(enabled=tools or ["search", "calculator"]),
            },
        ),
    }
    if max_tool_calls is not None:
        updates["governance"] = profile.governance.model_copy(
            update={"cost": CostProfile(max_tool_calls=max_tool_calls)},
        )
    return profile.model_copy(update=updates)


def _revision_from_layers(
    application: ApplicationEnvironmentProfile,
    layers: tuple[ProfileLayerInput, ...],
    *,
    store: InMemoryEffectiveProfileRevisionStore | None = None,
    predecessor: str | None = None,
) -> tuple[object, object]:
    resolution = resolve_profile(application, layers=layers)
    revision = materialize_effective_profile_revision(
        resolution,
        scope=_SCOPE,
        predecessor_revision_id=predecessor,
        store=store,
    )
    return resolution, revision


def test_revision_is_immutable() -> None:
    _, revision = _revision_from_layers(_application(), ())
    with pytest.raises(ValidationError):
        revision.revision_id = revision.revision_id  # type: ignore[misc]


def test_same_effective_profile_same_fingerprint_different_revision_ids() -> None:
    application = _application()
    layers = (
        ProfileLayerInput(
            layer=ProfileLayer.PLATFORM,
            delta=ProfileDelta(
                tool_profile=ProfileFieldUpdate(value=ToolProfile(enabled=["search"])),
            ),
        ),
    )
    resolution_a, revision_a = _revision_from_layers(application, layers)
    resolution_b, revision_b = _revision_from_layers(application, layers)
    assert resolution_a.fingerprint == resolution_b.fingerprint
    assert revision_a.fingerprint == revision_b.fingerprint
    assert revision_a.revision_id != revision_b.revision_id


def test_effective_diff_noop_for_same_semantics_different_provenance() -> None:
    application = _application(tools=["search", "shell"])
    layers_a = (
        ProfileLayerInput(
            layer=ProfileLayer.PLATFORM,
            delta=ProfileDelta(
                tool_profile=ProfileFieldUpdate(value=ToolProfile(enabled=["search"])),
            ),
        ),
        ProfileLayerInput(
            layer=ProfileLayer.EXECUTION,
            delta=ProfileDelta(
                tool_profile=ProfileFieldUpdate(value=ToolProfile(enabled=["search", "shell"])),
            ),
        ),
    )
    layers_b = (
        ProfileLayerInput(
            layer=ProfileLayer.PLATFORM,
            delta=ProfileDelta(
                tool_profile=ProfileFieldUpdate(value=ToolProfile(enabled=["search"])),
            ),
        ),
        ProfileLayerInput(
            layer=ProfileLayer.EXECUTION,
            delta=ProfileDelta(
                tool_profile=ProfileFieldUpdate(value=ToolProfile(enabled=["search", "calculator"])),
            ),
        ),
    )
    _, revision_a = _revision_from_layers(application, layers_a)
    _, revision_b = _revision_from_layers(application, layers_b)
    diff = diff_effective_profile_revisions(revision_a, revision_b)
    assert diff.is_empty
    assert revision_a.fingerprint == revision_b.fingerprint


def test_tools_added_removed_and_order_invariant() -> None:
    application = _application(tools=["search", "calculator"])
    _, revision_a = _revision_from_layers(
        application,
        (
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(value=ToolProfile(enabled=["search"])),
                ),
            ),
        ),
    )
    _, revision_b = _revision_from_layers(
        application,
        (
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(
                        value=ToolProfile(enabled=["calculator", "search"]),
                    ),
                ),
            ),
        ),
    )
    diff = diff_effective_profile_revisions(revision_a, revision_b)
    assert any(
        entry.path == "capabilities.tools.calculator"
        and entry.change_kind is ProfileDiffChangeKind.ADDED
        for entry in diff.entries
    )

    _, revision_c = _revision_from_layers(
        application,
        (
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(
                        value=ToolProfile(enabled=["search", "calculator"]),
                    ),
                ),
            ),
        ),
    )
    order_diff = diff_effective_profile_revisions(revision_b, revision_c)
    assert order_diff.is_empty


def test_cost_narrowing_visible_in_semantic_diff() -> None:
    application = _application(max_tool_calls=10)
    _, revision_a = _revision_from_layers(
        application,
        (
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(value=CostProfile(max_tool_calls=10)),
                ),
            ),
        ),
    )
    _, revision_b = _revision_from_layers(
        application,
        (
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(value=CostProfile(max_tool_calls=5)),
                ),
            ),
        ),
    )
    diff = diff_effective_profile_revisions(revision_a, revision_b)
    cost_entry = next(
        entry for entry in diff.entries if entry.path == "governance.cost.max_tool_calls"
    )
    assert cost_entry.before == "10"
    assert cost_entry.after == "5"
    assert cost_entry.change_kind is ProfileDiffChangeKind.NARROWED


def test_llm_provider_model_diff_paths() -> None:
    application = _application(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    _, revision_a = _revision_from_layers(application, ())
    _, revision_b = _revision_from_layers(
        _application(provider=LLMProvider.CLAUDE, model="claude-3-5-sonnet"),
        (),
    )
    diff = diff_effective_profile_revisions(revision_a, revision_b)
    paths = {entry.path for entry in diff.entries}
    assert "capabilities.llm.provider" in paths
    assert "capabilities.llm.model" in paths


def test_execution_mode_diff() -> None:
    _, revision_a = _revision_from_layers(_application(execution_mode=ExecutionMode.BALANCED), ())
    _, revision_b = _revision_from_layers(_application(execution_mode=ExecutionMode.STRICT), ())
    diff = diff_effective_profile_revisions(revision_a, revision_b)
    entry = next(entry for entry in diff.entries if entry.path == "meta.execution_mode")
    assert entry.before == ExecutionMode.BALANCED.value
    assert entry.after == ExecutionMode.STRICT.value


def test_historical_retrieval_and_append_only() -> None:
    store = InMemoryEffectiveProfileRevisionStore()
    application = _application()
    _, revision_a = _revision_from_layers(application, (), store=store)
    _, revision_b = _revision_from_layers(
        _application(max_tool_calls=5),
        (
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(value=CostProfile(max_tool_calls=5)),
                ),
            ),
        ),
        store=store,
        predecessor=revision_a.revision_id,
    )
    loaded_a = store.get(revision_a.revision_id, scope=_SCOPE)
    assert loaded_a is not None
    assert loaded_a.fingerprint == revision_a.fingerprint
    assert loaded_a.effective_profile == revision_a.effective_profile
    loaded_b = store.get(revision_b.revision_id, scope=_SCOPE)
    assert loaded_b is not None
    assert loaded_b.predecessor_revision_id == revision_a.revision_id
    assert loaded_a.revision_id != loaded_b.revision_id
    assert loaded_a.effective_profile.cost_profile.max_tool_calls != (
        loaded_b.effective_profile.cost_profile.max_tool_calls
    )


def test_execution_pinning_and_latest_revision_isolation() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    _, revision_a = _revision_from_layers(_application(), (), store=revision_store)
    _, revision_b = _revision_from_layers(
        _application(max_tool_calls=3),
        (
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(value=CostProfile(max_tool_calls=3)),
                ),
            ),
        ),
        store=revision_store,
    )
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision_a,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    binding = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
    )
    assert binding.revision_id == revision_a.revision_id
    assert binding.revision_id != revision_b.revision_id


def test_checkpoint_resume_preserves_pinned_revision() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    _, revision = _revision_from_layers(_application(), (), store=revision_store)
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        message="checkpoint task",
        context=TaskContext(capability="echo.basic"),
    )
    task = attach_revision_checkpoint_evidence_to_task(task, revision)
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id="tenant-a",
        resume_token="resume-1",
        task_state=task.state,
        task_snapshot=task.model_dump(mode="json"),
    )
    resumed_revision_id = revision_id_from_checkpoint(checkpoint)
    assert resumed_revision_id == revision.revision_id
    _, later_revision = _revision_from_layers(
        _application(max_tool_calls=1),
        (
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(value=CostProfile(max_tool_calls=1)),
                ),
            ),
        ),
        store=revision_store,
    )
    assert later_revision.revision_id != revision.revision_id
    binding = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
    )
    assert binding.revision_id == revision.revision_id


def test_background_redelivery_preserves_pinned_revision() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    _, revision = _revision_from_layers(_application(), (), store=revision_store)
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    first = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
    )
    second = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
    )
    assert first == second


def test_child_inherits_parent_pinned_revision() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    _, revision = _revision_from_layers(_application(), (), store=revision_store)
    parent_execution_id = mint_execution_id()
    child_execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision,
        tenant_id="tenant-a",
        execution_id=parent_execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    child_binding = inherit_child_execution_pinned_revision(
        tenant_id="tenant-a",
        parent_execution_id=parent_execution_id,
        child_execution_id=child_execution_id,
        pinning_store=pinning_store,
    )
    assert child_binding.revision_id == revision.revision_id


def test_missing_pinned_revision_fails_closed() -> None:
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    with pytest.raises(MissingPinnedEffectiveProfileRevisionError):
        require_execution_pinned_revision(
            tenant_id="tenant-a",
            execution_id=ExecutionId("exec_00000000000000000000000000000001"),
            pinning_store=pinning_store,
        )


def test_resolve_revision_for_execution_fails_closed_when_store_missing_revision() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    _, revision = _revision_from_layers(_application(), ())
    execution_id = mint_execution_id()
    pinning_store.pin(
        EffectiveProfileExecutionBinding(
            tenant_id="tenant-a",
            execution_id=execution_id,
            revision_id=revision.revision_id,
            fingerprint=revision.fingerprint,
        )
    )
    with pytest.raises(MissingPinnedEffectiveProfileRevisionError):
        resolve_revision_for_execution(
            tenant_id="tenant-a",
            execution_id=execution_id,
            pinning_store=pinning_store,
            revision_store=revision_store,
            scope_application_id=_SCOPE.application_id,
            scope_tenant_id=_SCOPE.tenant_id,
        )


def test_build_harness_host_runtime_materializes_effective_profile_revision() -> None:
    manifest = ApplicationManifest.lab(
        app_id="revision_host",
        name="Revision Host",
        route_prefix="/v1/revision_host",
        env_prefix="REVISION_HOST_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    configured = ApplicationEnvironmentProfile.lab_defaults(profile_id="revision_host.lab")
    store = InMemoryEffectiveProfileRevisionStore()
    runtime = build_harness_host_runtime(
        manifest,
        configured,
        tenant_id="tenant-a",
        use_in_memory_trace=True,
        revision_store=store,
    )
    assert runtime.effective_profile_revision is not None
    assert runtime.profile_resolution is not None
    assert runtime.effective_profile_revision.fingerprint == runtime.profile_resolution.fingerprint
    loaded = store.get(
        runtime.effective_profile_revision.revision_id,
        scope=EffectiveProfileRevisionScope(application_id="revision_host", tenant_id="tenant-a"),
    )
    assert loaded == runtime.effective_profile_revision
