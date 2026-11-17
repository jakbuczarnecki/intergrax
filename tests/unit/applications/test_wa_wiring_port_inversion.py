# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-G (PT-WA): Tier-3 builds, Tier-1 applies, through neutral ports (NT-WA-05..08)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from intergrax.applications._shared import host_orchestration_backend_spec_builder as tier3_builder
from intergrax.applications._shared.application_host_orchestration_wiring import (
    build_application_host_orchestration_wiring,
)
from intergrax.applications._shared.host_execution_capacity_policy import (
    HostExecutionCapacityPolicyError,
)
from intergrax.applications._shared.host_orchestration_backend_spec_builder import (
    build_host_orchestration_loop_init_spec_from_environment,
)
from intergrax.applications._shared.host_orchestration_environment_resolution import (
    LiveEnvironmentGraphTopologySource,
    host_orchestration_planning_spec,
)
from intergrax.applications._shared.security_assembly_resolver import SecurityAssemblyError
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec, GraphNode
from intergrax.runtime.execution.environment_orchestration_materialization import (
    EnvironmentOrchestrationMaterialization,
    materialize_host_orchestration_backend,
)
from intergrax.runtime.execution.host_orchestration_assembly_validation import (
    assert_host_orchestration_application_assembly,
)
from intergrax.runtime.execution.host_orchestration_planner_classifier_wiring import (
    GraphSpecSeedingPlanner,
    resolve_nexus_task_planner,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext

from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def test_wa_05_strict_assembly_refusal_through_port() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="wa.05.strict")
    env.execution_mode = ExecutionMode.STRICT
    wiring = build_application_host_orchestration_wiring(env)
    assert "TenantSecurityMiddleware" in wiring.security_wiring.enabled_middleware
    bare_host = SimpleNamespace(
        middleware=SimpleNamespace(registered_middleware_names=lambda: frozenset()),
    )
    materialization = EnvironmentOrchestrationMaterialization(_backend=bare_host)  # type: ignore[arg-type]
    with pytest.raises(SecurityAssemblyError, match="TenantSecurityMiddleware"):
        assert_host_orchestration_application_assembly(materialization, wiring=wiring)


def test_wa_06_nonconforming_wiring_object_refused() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="wa.06.port")
    registry = _registry()
    spec = build_host_orchestration_loop_init_spec_from_environment(registry, env=env)

    class _PartialWiring:
        """Has ``apply`` but not ``assert_assembly`` / ``apply_runtime_wiring``."""

        def __init__(self) -> None:
            self.applied = 0

        def apply(self, target: object) -> None:
            self.applied += 1

    partial = _PartialWiring()
    broken = dataclasses.replace(spec, application_wiring=partial)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="HostOrchestrationApplicationWiringPort"):
        materialize_host_orchestration_backend(registry, broken)
    assert partial.applied == 0


def test_wa_07_strict_capacity_refusal_precedes_ee(monkeypatch: pytest.MonkeyPatch) -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="wa.07.capacity").model_copy(
        update={"orchestration_profile": OrchestrationProfile(long_running_enabled=True)},
    )
    assert env.orchestration_profile.max_parallel_nodes is None
    calls: list[object] = []

    def _spy(*args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        raise AssertionError("Tier-1 builder must not be reached")

    monkeypatch.setattr(tier3_builder, "build_host_orchestration_loop_init_spec", _spy)
    with pytest.raises(HostExecutionCapacityPolicyError, match="max_parallel_nodes"):
        build_host_orchestration_loop_init_spec_from_environment(_registry(), env=env)
    assert calls == []


def test_wa_08_planner_reads_live_graph_topology() -> None:
    first = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id="EchoAgent")],
        trigger_capabilities=["wa.first"],
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="wa.08.live").model_copy(
        update={"graph_spec": first},
    )
    source = LiveEnvironmentGraphTopologySource(env)
    assert source.current_graph_topology() is first
    planner = resolve_nexus_task_planner(host_orchestration_planning_spec(env))
    assert isinstance(planner, GraphSpecSeedingPlanner)

    registry = _registry()
    task_first = Task(
        tenant_id="t",
        user_id="u",
        message="m",
        context=TaskContext(capability="wa.first"),
    )
    assert planner.plan(task_first, registry).plan_metadata["planner_source"] == "graph_spec"

    env.graph_spec = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id="EchoAgent")],
        trigger_capabilities=["wa.second"],
    )
    assert source.current_graph_topology() is env.graph_spec
    task_second = Task(
        tenant_id="t",
        user_id="u",
        message="m",
        context=TaskContext(capability="wa.second"),
    )
    assert planner.plan(task_second, registry).plan_metadata["planner_source"] == "graph_spec"
    # The first capability is no longer a seed trigger: the live topology replaced the old one.
    assert "planner_source" not in planner.plan(task_first, registry).plan_metadata
