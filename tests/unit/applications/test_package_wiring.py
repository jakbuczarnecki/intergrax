# © Artur Czarnecki. All rights reserved.

"""APP-EVOL-7 — ApplicationPackage wiring."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.package_wiring import (
    _validate_graph_spec_capabilities,
    build_application_package,
    collect_application_dependencies,
    compute_package_checksum,
    package_id_for_app,
    validate_application_package_closure,
)
from intergrax.applications._shared.product_manifest_registry import iter_strict_product_manifests
from intergrax.applications.contracts.application_package import ApplicationDependencyKind
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec, GraphNode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_package_id_for_app() -> None:
    assert package_id_for_app("legal") == "com.intergrax.legal"


def test_collect_dependencies_includes_agent_and_profile() -> None:
    manifest = ApplicationManifest.lab(
        app_id="pkg_demo",
        name="Pkg Demo",
        route_prefix="/v1/pkg_demo",
        env_prefix="PKG_DEMO_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="pkg_demo.scaffold")
    deps = collect_application_dependencies(manifest, env)
    kinds = {dep.kind for dep in deps}
    assert ApplicationDependencyKind.AGENT in kinds
    assert ApplicationDependencyKind.PROFILE_FRAGMENT in kinds


@pytest.mark.no_ci
def test_strict_product_package_closure_passes() -> None:
    product_id, manifest = next(iter(iter_strict_product_manifests()))
    env = manifest.resolved_environment()
    wiring = wire_application_environment(manifest, env, conformance_check=False)
    package = build_application_package(manifest, env)
    violations = validate_application_package_closure(
        package,
        manifest,
        env,
        wiring.registry_snapshot,
        capability_graph=wiring.capability_graph,
    )
    assert violations == [], f"{product_id}: {violations}"


def test_graph_trigger_capability_not_required_on_roster() -> None:
    graph = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id="EchoAgent")],
        trigger_capabilities=["echo.orchestration.pipeline"],
    )
    manifest = ApplicationManifest.lab(
        app_id="graph_trigger_demo",
        name="Graph Trigger Demo",
        route_prefix="/v1/graph_trigger_demo",
        env_prefix="GRAPH_TRIGGER_DEMO_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="graph_trigger_demo.scaffold").model_copy(
        update={"graph_spec": graph},
    )
    assert _validate_graph_spec_capabilities(manifest, env) == []


def test_graph_invalid_node_still_fails_closure() -> None:
    graph = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id="missing_agent")],
        trigger_capabilities=["echo.orchestration.pipeline"],
    )
    manifest = ApplicationManifest.lab(
        app_id="graph_invalid_node_demo",
        name="Graph Invalid Node Demo",
        route_prefix="/v1/graph_invalid_node_demo",
        env_prefix="GRAPH_INVALID_NODE_DEMO_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="graph_invalid_node_demo.scaffold").model_copy(
        update={"graph_spec": graph},
    )
    violations = _validate_graph_spec_capabilities(manifest, env)
    assert any("not found on manifest roster" in violation for violation in violations)


def test_package_checksum_is_stable() -> None:
    manifest = ApplicationManifest.lab(
        app_id="checksum_demo",
        name="Checksum Demo",
        route_prefix="/v1/checksum_demo",
        env_prefix="CHECKSUM_DEMO_",
        agents=[AgentBinding.mount(EchoAgent)],
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="checksum_demo.scaffold")
    package = build_application_package(manifest, env)
    assert package.distribution.checksum == compute_package_checksum(
        package.model_copy(
            update={
                "distribution": package.distribution.model_copy(update={"checksum": ""}),
            },
        ),
    )
