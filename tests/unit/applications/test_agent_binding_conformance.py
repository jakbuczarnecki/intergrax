# © Artur Czarnecki. All rights reserved.

"""AgentBinding identity conformance — fail-closed roster validation (PLATFORM-5B-R1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.environment_conformance import (
    EnvironmentSkillToolConsistencyCheck,
)
from intergrax.applications._shared.package_wiring import (
    _validate_graph_spec_capabilities,
    collect_application_dependencies,
)
from intergrax.applications.contracts.application_package import ApplicationDependencyKind
from intergrax.applications.contracts.agent_ref import qualname_for_agent
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.errors import ApplicationManifestConformanceError
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec, GraphNode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _lab_env() -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="binding_conformance.lab")
    skill_profile = env.skill_profile.model_copy(update={"enabled": ["harness.tool_smoke"]})
    return env.model_copy(
        update={
            "capabilities": env.capabilities.model_copy(update={"skills": skill_profile}),
        },
    )


def test_agent_type_binding_passes_conformance() -> None:
    binding = AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])
    violations = EnvironmentSkillToolConsistencyCheck(fail_on_violation=False).validate_binding(
        binding,
        _lab_env(),
    )
    assert violations == []


def test_import_path_binding_passes_conformance() -> None:
    binding = AgentBinding.deserialize(
        import_path=qualname_for_agent(EchoAgent),
        capabilities=["echo.basic"],
    )
    violations = EnvironmentSkillToolConsistencyCheck(fail_on_violation=False).validate_binding(
        binding,
        _lab_env(),
    )
    assert violations == []


def test_contract_reference_binding_passes_conformance() -> None:
    binding = AgentBinding.reference("echo")
    violations = EnvironmentSkillToolConsistencyCheck(fail_on_violation=False).validate_binding(
        binding,
        _lab_env(),
    )
    assert violations == []


def test_harness_reference_manifest_contract_binding_passes() -> None:
    from intergrax.applications.reference.harness_manifest_catalog import build_harness_reference_manifests

    manifest = build_harness_reference_manifests()[0]
    binding = manifest.enabled_agents()[0]
    violations = EnvironmentSkillToolConsistencyCheck(fail_on_violation=False).validate_binding(
        binding,
        manifest.resolved_environment(),
    )
    assert violations == []


def test_unresolvable_binding_fails_closed() -> None:
    binding = AgentBinding.model_construct(enabled=True)
    check = EnvironmentSkillToolConsistencyCheck(fail_on_violation=False)
    violations = check.validate_binding(binding, _lab_env())
    assert len(violations) == 1
    assert "no resolvable agent identity" in violations[0]

    with pytest.raises(ApplicationManifestConformanceError, match="no resolvable agent identity"):
        EnvironmentSkillToolConsistencyCheck(fail_on_violation=True).validate_binding(
            binding,
            _lab_env(),
        )


def test_blank_contract_id_rejected_at_construction() -> None:
    with pytest.raises(ValidationError, match="contract_id"):
        AgentBinding.reference("   ")


def test_graph_spec_unknown_agent_fails_closure() -> None:
    graph = ApplicationGraphSpec(nodes=[GraphNode(agent_id="missing_agent")])
    manifest = ApplicationManifest.lab(
        app_id="graph_roster_gate",
        name="Graph Roster Gate",
        route_prefix="/v1/graph_roster_gate",
        env_prefix="GRAPH_ROSTER_GATE_",
        agents=[AgentBinding.reference("echo")],
    )
    env = _lab_env().model_copy(update={"graph_spec": graph})
    violations = _validate_graph_spec_capabilities(manifest, env)
    assert any("not found on manifest roster" in violation for violation in violations)


def test_contract_reference_package_closure_uses_contract_id() -> None:
    manifest = ApplicationManifest.lab(
        app_id="contract_ref_pkg",
        name="Contract Ref Package",
        route_prefix="/v1/contract_ref_pkg",
        env_prefix="CONTRACT_REF_PKG_",
        agents=[AgentBinding.reference("incident_investigator")],
    )
    env = _lab_env()
    deps = collect_application_dependencies(manifest, env)
    agent_deps = [dep for dep in deps if dep.kind is ApplicationDependencyKind.AGENT]
    assert len(agent_deps) == 1
    assert agent_deps[0].ref == "incident_investigator"
    assert agent_deps[0].version_constraint == "*"
