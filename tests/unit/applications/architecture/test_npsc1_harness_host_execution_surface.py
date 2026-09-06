# © Artur Czarnecki. All rights reserved.

"""NPSC-1: HarnessHostRuntime public execution surface closure gate."""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    build_harness_host_runtime,
)
from intergrax.applications._shared.harness_host_runtime_compat import (
    resolve_harness_host_nexus_loop_legacy,
)
from intergrax.applications._shared.production_platform_persistence import (
    build_reference_production_platform_persistence,
    resolve_reference_production_strict_host_environment,
)
from intergrax.agent_distribution.roster import EffectiveRosterEntry
from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
from intergrax.applications._shared.registry_projection import (
    MaterializedRegistryProjection,
    RegistryProjectionInputBundle,
    build_registry_projection,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from tests.unit.applications.test_registry_projection_ap10 import (
    ECHO_BUILDERS,
    _ARTIFACT,
    _ENV,
    _ROSTER_A,
    _entry,
    _manifest,
    _revision,
    _roster,
    _resolver_from_roster,
)
from intergrax.contracts.execution_identity import (
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
    validate_execution_id,
)
from intergrax.runtime.execution.facade import Execution as ExecutionFacade
from intergrax.runtime.execution.host_task import HostTaskExecution, HostTaskExecutionPort
from intergrax.runtime.execution.request import ExecutionCapability
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.task_adapter import TaskExecutionInput
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from research_application.tests.research_ac3_projection import build_research_test_registry_projection
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_environment_profile

from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HOST_TASK_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "host_task.py"


@pytest.fixture(autouse=True)
def _relax_harness_environment_assertions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET",
        "unit-test-diagnostic-problem-list-cursor-secret",
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.package_wiring.assert_manifest_package_closure",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.environment_wiring.assert_application_owned_tool_conformance",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.diagnostic_assembly_resolver.assert_diagnostic_assembly_valid",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.harness_host_runtime.assert_observability_assembly_valid",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.environment_wiring.validate_capability_dependencies_for_environment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "intergrax.runtime.nexus.nexus_loop.validate_durable_attempt_lifecycle_for_composition",
        lambda **_kwargs: None,
    )


def _platform_persistence_kwargs() -> dict[str, object]:
    platform = build_reference_production_platform_persistence()
    return {
        "key_value_cache": platform.kv_store,
        "document_store": platform.document_store,
    }


def _strict_environment():
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    return ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)


def _build_projection(
    *,
    roster_entries: tuple[EffectiveRosterEntry, ...],
) -> MaterializedRegistryProjection:
    roster = _roster(roster_entries)
    manifest = _manifest()
    revision = _revision(
        "rev-npsc1",
        roster_revision_id=roster.effective_roster_revision_id or _ROSTER_A,
    )
    bundle = RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=roster,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        factory_resolver=_resolver_from_roster(roster),
        builders=ECHO_BUILDERS,
        materialization_artifact_digest=_ARTIFACT,
    )
    return build_registry_projection(bundle)


def _build_echo_runtime() -> HarnessHostRuntime:
    manifest = _manifest()
    environment = resolve_reference_production_strict_host_environment(_strict_environment())
    projection = _build_projection(roster_entries=(_entry("search"),))
    return build_harness_host_runtime(
        manifest,
        environment,
        registry_projection=projection,
        registry_assembly_mode=RegistryAssemblyMode.REVISION_BOUND,
        **_platform_persistence_kwargs(),
    )


def _build_research_runtime() -> HarnessHostRuntime:
    settings = ResearchBackendSettings.from_env()
    env = resolve_reference_production_strict_host_environment(
        build_research_environment_profile(settings),
    )
    projection = build_research_test_registry_projection(settings)
    return build_harness_host_runtime(
        RESEARCH_APPLICATION_MANIFEST.model_copy(update={"environment": env}),
        env,
        settings=settings,
        registry_projection=projection,
        **_platform_persistence_kwargs(),
    )


def test_harness_host_runtime_exposes_canonical_execution_surface() -> None:
    runtime = _build_echo_runtime()
    assert isinstance(runtime.execution, HostTaskExecution)


def test_harness_host_runtime_public_fields_do_not_include_nexus_loop() -> None:
    public_fields = {field.name for field in fields(HarnessHostRuntime)}
    assert "nexus_loop" not in public_fields
    assert "execution" in public_fields


def test_host_task_execution_port_has_no_nexus_loop() -> None:
    source = inspect.getsource(HostTaskExecutionPort)
    assert "nexus_loop" not in source
    assert "NexusLoop" not in source


def test_host_task_execution_has_no_public_nexus_loop_property() -> None:
    public_names = {
        name
        for name in dir(HostTaskExecution)
        if not name.startswith("_") and name != "execute"
    }
    assert "nexus_loop" not in public_names
    assert not hasattr(HostTaskExecution, "nexus_loop")


def test_host_task_module_has_no_nexus_import() -> None:
    text = _HOST_TASK_PATH.read_text(encoding="utf-8")
    assert "NexusLoop" not in text
    assert "nexus_loop" not in text


def test_runtime_execution_nexus_loop_is_impossible() -> None:
    runtime = _build_echo_runtime()
    assert not hasattr(runtime.execution, "nexus_loop")


def test_legacy_compat_resolves_internal_nexus_without_public_field() -> None:
    runtime = _build_echo_runtime()
    nexus_loop = resolve_harness_host_nexus_loop_legacy(runtime)
    assert isinstance(nexus_loop, NexusLoop)
    assert not hasattr(runtime.execution, "nexus_loop")


@pytest.mark.asyncio
async def test_agentic_execution_does_not_require_caller_nexus() -> None:
    runtime = _build_echo_runtime()
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="agentic proof",
        context=TaskContext(capability="echo.basic"),
        agent_id="search",
    )
    nexus_loop = resolve_harness_host_nexus_loop_legacy(runtime)

    with patch.object(nexus_loop, "handle_task", new_callable=AsyncMock) as handle_task_mock:
        with patch(
            "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
            new_callable=AsyncMock,
            return_value=TaskResult(
                task_id=task.task_id,
                run_id=mint_run_id(),
                state=TaskState.COMPLETED,
                answer="ok",
            ),
        ):
            await runtime.execution.execute(task)

    handle_task_mock.assert_not_called()


@pytest.mark.asyncio
async def test_orchestration_execution_reaches_internal_nexus_backend() -> None:
    runtime = _build_research_runtime()
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="orchestration proof",
        context=TaskContext(capability="research.pipeline"),
    )
    nexus_loop = resolve_harness_host_nexus_loop_legacy(runtime)

    with patch.object(
        nexus_loop,
        "handle_task",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=task.task_id,
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="pipeline",
        ),
    ) as handle_task_mock:
        await runtime.execution.execute(task)

    handle_task_mock.assert_called_once()


@pytest.mark.asyncio
async def test_public_execution_path_has_single_root_lifecycle() -> None:
    runtime = _build_echo_runtime()
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="identity proof",
        context=TaskContext(capability="echo.basic"),
        agent_id="search",
    )
    facade_calls = 0
    router_calls = 0
    original_execute = ExecutionFacade.execute

    async def _count_facade_execute(self, request, *, options):
        nonlocal facade_calls
        facade_calls += 1
        return await original_execute(self, request, options=options)

    async def _count_router_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request,
    ) -> TaskResult:
        nonlocal router_calls
        router_calls += 1
        execution_id = require_active_execution_id()
        validate_execution_id(execution_id)
        return TaskResult(
            task_id=task.task_id,
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="one",
        )

    with patch.object(ExecutionFacade, "execute", _count_facade_execute):
        with patch.object(StrategyExecutionRouter, "execute", _count_router_execute):
            await runtime.execution.execute(task)

    assert facade_calls == 1
    assert router_calls == 1


def test_research_orchestration_capability_resolves_without_caller_nexus() -> None:
    settings = ResearchBackendSettings.from_env()
    env = build_research_environment_profile(settings)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="strategy proof",
        context=TaskContext(capability="research.pipeline"),
    )
    from intergrax.runtime.execution.host_task import resolve_task_execution_capabilities
    from intergrax.runtime.execution.task_adapter import execution_request_from_task
    from intergrax.runtime.nexus.orchestration_capabilities import orchestration_capabilities_from_triggers

    graph_spec = env.graph_spec
    orchestration_triggers = orchestration_capabilities_from_triggers(
        graph_spec.trigger_capabilities if graph_spec is not None else None,
    )
    pipeline_suffix = (
        graph_spec.pipeline_capability_suffix if graph_spec is not None else ".pipeline"
    )
    capabilities = resolve_task_execution_capabilities(
        task,
        orchestration_triggers=orchestration_triggers,
        pipeline_capability_suffix=pipeline_suffix,
    )
    request = execution_request_from_task(
        task,
        capabilities=capabilities,
        output_type=TaskResult,
    )
    assert StrategyResolver().resolve(request) is ExecutionStrategy.ORCHESTRATION
    assert capabilities == frozenset({ExecutionCapability.ORCHESTRATION})
