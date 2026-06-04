# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec, GraphEdge, GraphNode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile


@pytest.mark.gate
def test_environment_profile_json_roundtrip() -> None:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="roundtrip.test")
    restored = ApplicationEnvironmentProfile.model_validate_json(profile.model_dump_json())
    assert restored.profile_id == profile.profile_id
    assert restored.spec_version == profile.spec_version


@pytest.mark.gate
def test_manifest_json_roundtrip() -> None:
    from echo.echo_agent import EchoAgent

    manifest = ApplicationManifest.lab(
        app_id="rt",
        name="Roundtrip",
        route_prefix="/v1/rt",
        environment=ApplicationEnvironmentProfile.lab_defaults(profile_id="rt"),
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
        integration_profile=IntegrationProfile.lab_stack(),
    )
    restored = ApplicationManifest.model_validate_json(manifest.model_dump_json())
    assert restored.app_id == "rt"
    assert len(restored.agents) == 1


@pytest.mark.gate
def test_graph_spec_json_roundtrip() -> None:
    spec = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id="echo", contract_id="echo")],
        edges=[GraphEdge(source_agent_id="echo", target_agent_id="echo")],
    )
    restored = ApplicationGraphSpec.model_validate_json(spec.model_dump_json())
    assert restored.nodes[0].agent_id == "echo"
