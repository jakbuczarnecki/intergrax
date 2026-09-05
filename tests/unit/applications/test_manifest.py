# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from echo.echo_agent import EchoAgent
from intergrax.applications.contracts.agent_ref import qualname_for_agent
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit


def test_agent_binding_mount_requires_explicit_contract_id() -> None:
    with pytest.raises(TypeError):
        AgentBinding.mount(EchoAgent)  # type: ignore[call-arg]


def test_agent_binding_mount_sets_type_and_import_path() -> None:
    binding = AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])
    assert binding.agent_type is EchoAgent
    assert binding.import_path == qualname_for_agent(EchoAgent)


def test_agent_binding_mount_with_factory() -> None:
    def build_echo(_ctx, _binding) -> EchoAgent:
        return EchoAgent()

    binding = AgentBinding.mount(EchoAgent, contract_id="echo", factory=build_echo)
    assert binding.factory is build_echo
    assert binding.factory_path is not None


def test_agent_binding_rejects_factory_and_builder_key_on_mount() -> None:
    with pytest.raises(ValueError, match="factory or builder_key"):
        AgentBinding.mount(EchoAgent, contract_id="echo", factory=lambda _c, _b: EchoAgent(), builder_key="echo")


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "EchoAgent",
        "echo.EchoAgent",
        "echo.echo_agent.echo_agent",
    ],
)
def test_agent_binding_deserialize_rejects_bad_import_path(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        AgentBinding.deserialize(import_path=bad_path)


def test_application_manifest_lab_factory() -> None:
    manifest = ApplicationManifest.lab(
        app_id="my_lab",
        name="My Lab",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    assert manifest.app_id == "my_lab"
    assert manifest.agents[0].agent_type is EchoAgent
    assert len(manifest.enabled_agents()) == 1


def test_application_manifest_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ApplicationManifest.lab(
            app_id="x",
            name="X",
            agents=[AgentBinding.mount(EchoAgent, contract_id="echo")],
            unknown_field=True,
        )
