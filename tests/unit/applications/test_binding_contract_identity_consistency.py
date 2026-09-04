# © Artur Czarnecki. All rights reserved.

"""Cross-component contract identity consistency (projection · graph · package · deploy)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.capability_graph_deploy_gate import (
    validate_strict_capability_graph_deploy,
)
from intergrax.applications._shared.capability_graph_wiring import (
    build_environment_seed_capability_graph,
    resolve_environment_capability_graph,
)
from intergrax.applications._shared.package_wiring import (
    build_application_package,
    validate_application_package_closure,
)
from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications.contracts.application_capability_projection import (
    application_capability_descriptor_from_manifest,
    resolve_binding_contract_id,
)
from intergrax.applications.contracts.application_package import ApplicationDependencyKind
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CONTRACT_ID = "agent.foo"


def _reference_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="identity_consistency",
        name="Identity Consistency",
        route_prefix="/v1/identity_consistency",
        env_prefix="IDENTITY_CONSISTENCY_",
        agents=[AgentBinding.reference(_CONTRACT_ID)],
    )


def _empty_snapshot() -> HarnessRegistrySnapshot:
    return HarnessRegistrySnapshot(
        integration_profile=IntegrationProfile(),
        tool_registry=None,
        skill_registry=None,
        agent_registry=None,
        prompt_registry=None,
        policy_bundle=None,
    )


def test_manifest_binding_contract_id_consistent_across_consumers() -> None:
    manifest = _reference_manifest()
    contract_id = resolve_binding_contract_id(manifest.enabled_agents()[0])

    descriptor = application_capability_descriptor_from_manifest(manifest)
    assert descriptor.agent_contract_ids == (contract_id,)

    snapshot = _empty_snapshot()
    seed_graph = build_environment_seed_capability_graph(manifest, snapshot)
    agent_node = f"agent:{contract_id}"
    assert any(node.node_id == agent_node for node in seed_graph.nodes)

    view = resolve_environment_capability_graph(manifest, ApplicationEnvironmentProfile.lab_defaults(), snapshot)
    assert view.contains_node(agent_node)

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="identity_consistency.scaffold")
    package = build_application_package(manifest, env)
    agent_refs = {
        dep.ref for dep in package.dependencies if dep.kind is ApplicationDependencyKind.AGENT
    }
    assert contract_id in agent_refs

    violations = validate_application_package_closure(
        package,
        manifest,
        env,
        snapshot,
        capability_graph=view,
    )
    assert violations == []

    deploy_result = validate_strict_capability_graph_deploy(view, snapshot, manifest, env)
    assert not any(f"roster agent {contract_id!r} missing" in error for error in deploy_result.errors)
