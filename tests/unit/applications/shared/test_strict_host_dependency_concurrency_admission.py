# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R1 — strict admission configuration authority (negative matrix)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.strict_host_dependency_concurrency_admission import (
    build_dependency_concurrency_admission_for_tool_ids,
    build_reference_host_tool_dependency_concurrency_admission,
    resolve_dependency_concurrency_admission_configuration,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.presets import (
    lab_reference_tool_profile,
)
from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionConfiguration,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPolicy,
    DependencyConcurrencyPolicyBinding,
)
from intergrax.runtime.resilience.dependency_attempt_boundary_composition import (
    ToolDependencyAttemptBoundaryMaterializationError,
    materialize_tool_dependency_attempt_boundary,
)
from intergrax.tools.contracts.tool_profile import ToolProfile
from testing_support.dependency_concurrency_admission_config import (
    tool_dependency_concurrency_admission_configuration,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_strict_missing_configuration_fails_closed_at_materialization() -> None:
    resolved = resolve_dependency_concurrency_admission_configuration(
        None,
        production_mode=True,
    )
    assert resolved is None
    with pytest.raises(ToolDependencyAttemptBoundaryMaterializationError):
        materialize_tool_dependency_attempt_boundary(resolved, production_mode=True)


def test_strict_explicit_configuration_preserved() -> None:
    explicit = tool_dependency_concurrency_admission_configuration(
        "probe.tool",
        max_concurrent_calls=3,
    )
    resolved = resolve_dependency_concurrency_admission_configuration(
        explicit,
        production_mode=True,
    )
    assert resolved is explicit
    boundary = materialize_tool_dependency_attempt_boundary(resolved, production_mode=True)
    assert boundary is not None
    boundary.close()


def test_resolve_does_not_synthesize_for_unlisted_registry_tool() -> None:
    synthesized = build_dependency_concurrency_admission_for_tool_ids(("listed.tool",))
    resolved = resolve_dependency_concurrency_admission_configuration(
        synthesized,
        production_mode=True,
    )
    assert resolved is not None
    assert all(
        binding.identity.value == "listed.tool"
        for binding in resolved.bindings
        if binding.identity.kind is DependencyConcurrencyKind.TOOL
    )
    assert not any(
        binding.identity.value == "unlisted.new.tool"
        for binding in resolved.bindings
    )


def test_reference_host_platform_defaults_materializes_tool_admission() -> None:
    product = ApplicationEnvironmentProfile.product_defaults(
        profile_id="prod_q_r1.reference",
        skill_bundles=["harness"],
    )
    tools = lab_reference_tool_profile(harness_tools=False).model_copy(
        update={"enabled": ["read_file"]},
    )
    base = product.model_copy(
        update={
            "capabilities": product.capabilities.model_copy(update={"tools": tools}),
        },
    )
    env = base.with_reference_host_platform_defaults()
    admission = env.reliability_profile.dependency_concurrency_admission
    assert admission is not None
    tool_bindings = [
        binding
        for binding in admission.bindings
        if binding.identity.kind is DependencyConcurrencyKind.TOOL
    ]
    assert tool_bindings
    assert all(
        binding.policy.max_concurrent_calls == env.orchestration_profile.max_inflight_nodes
        for binding in tool_bindings
    )
    boundary = materialize_tool_dependency_attempt_boundary(admission, production_mode=True)
    assert boundary is not None
    boundary.close()


def test_declarative_strict_host_fail_closed_without_profile_admission() -> None:
    from intergrax.applications._shared.declarative_tool_wiring import (
        build_declarative_invoker_for_application_host,
    )
    from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
    from intergrax.applications._shared.wiring import build_application_registry
    from intergrax.applications._shared.application_composition_context import (
        composition_for_factory_context,
    )
    from intergrax.applications.contracts.build_context import ApplicationBuildContext
    from intergrax.applications.contracts.execution_mode import ExecutionMode
    from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
    from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
    from intergrax.tools.registry import ToolRegistry
    from intergrax.tools.registry.wiring import ToolWiringContext
    from echo.echo_agent import EchoAgent

    manifest = ApplicationManifest.lab(
        app_id="decl_admission_fail_closed",
        name="Decl Admission",
        route_prefix="/v1/decl_admission",
        env_prefix="DECL_ADMISSION_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    build_ctx = ApplicationBuildContext.for_manifest(manifest)
    composition = composition_for_factory_context(
        build_ctx,
        policy_bundle=RuntimePolicyBundle(),
    )
    registry = build_application_registry(manifest, build_ctx, composition=composition)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="decl.strict").model_copy(
        update={"execution_mode": ExecutionMode.STRICT},
    )
    assert env.reliability_profile.dependency_concurrency_admission is None
    wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=["read_file"]),
        wiring_context=ToolWiringContext(),
        registry=ToolRegistry(),
    )
    with pytest.raises(ToolDependencyAttemptBoundaryMaterializationError):
        build_declarative_invoker_for_application_host(
            wiring,
            env,
            manifest=manifest,
            agent_registry=registry,
            tenant_id="tenant-a",
        )


def test_empty_tool_profile_yields_no_reference_admission() -> None:
    assert (
        build_reference_host_tool_dependency_concurrency_admission(
            ToolProfile(),
            max_concurrent_calls=8,
        )
        is None
    )
